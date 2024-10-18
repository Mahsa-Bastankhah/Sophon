import random
import numpy as np
from torch.utils.data import DataLoader
import os
from datetime import datetime
import argparse
import json
import sys
from CustomeDataset import CustomDataset
DIM_SIGNATURE=256
DIM_HASH=24

sys.path.append('../')
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--false_rate', default=0.5, type=float, help='fraction of target dataset to include in natural loop')
    parser.add_argument('--bs', default=150, type=int)
    parser.add_argument('--ml_loop', default=1, type=int)
    parser.add_argument('--nl_loop', default=1, type=int)
    parser.add_argument('--total_loop', default=1000, type=int)
    parser.add_argument('--alpha', default=3.0, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=5.0, type=float, help='coefficient of natural lr')
    parser.add_argument('--test_iterval', default=10, type=int)
    parser.add_argument('--arch', default='caformer', type=str)
    parser.add_argument('--gpus', default='0,1', type=str)
    parser.add_argument('--dataset', default='', type=str, choices=['CIFAR10', 'MNIST', 'SVHN', 'STL', 'CINIC'])
    parser.add_argument('--finetune_epochs', default=1, type=int)
    parser.add_argument('--truly_finetune_epochs', default=20, type=int)
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--fast_lr', default=0.0001, type=float)
    parser.add_argument('--root', default='results', type=str) 
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--partial', default='no', type=str, help='whether only use last ten batch to maml')
    parser.add_argument('--adaptation_steps', default=50, type=int) ## number of full batches used in the inner finetuning
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    args = parser.parse_args()
    return args
args = args_parser()
# if args.gpus: 
#     gpu_list = args.gpus.split(',')
#     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
#     devices_id = [id for id in range(len(gpu_list))]
from utils import save_bn, load_bn, check_gradients, accuracy, get_pretrained_model, test_original, test, initialize00, set_seed, save_data, get_finetuned_model, initialize, get_new_model, get_input
from tqdm import tqdm
import torch
from torch import nn, optim
from utils import get_dataset
import wandb
import learn2learn as l2l
import copy
import timm

import GPUtil

def select_least_busy_gpus(num_gpus_needed=2):
    # Get list of available GPUs sorted by least memory usage
    available_gpus = GPUtil.getAvailable(order='memory', limit=num_gpus_needed)

    if len(available_gpus) < num_gpus_needed:
        raise RuntimeError(f"Only {len(available_gpus)} GPUs are available, but {num_gpus_needed} are required.")
    
    # Convert GPU ids to a string format expected by CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(map(str, available_gpus))
    print(f"Assigning to least busy GPUs: {gpu_list}")
    
    # Set the environment variable to limit visible GPUs to the least busy ones
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

# Automatically select 2 least busy GPUs (or however many you need)
select_least_busy_gpus(num_gpus_needed=2)

 # Define a function to test accuracy on the target test dataset
def test_target(model, target_testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, signatures, hash_x, targets, false_flag in target_testloader:
            inputs, signatures, hash_x, targets = inputs.to(device), signatures.to(device), hash_x.to(device), targets.to(device)
            
            # Creating combined input using the `get_input` function
            combined_input = get_input(inputs, signatures, hash_x, INPUT_RESOLUTION=32**2)
            
            # Forward pass
            outputs = model(combined_input)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total

    return accuracy
def fast_adapt_multibatch(batches, learner, loss, shots, ways, device):
    # Adapt the model
    learner = initialize(args, learner)
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    print(len(batches))
    for index,batch in enumerate(batches):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        # adaptation_indices[np.arange(shots*ways)] = True
        adaptation_indices[np.random.choice(np.arange(data.size(0)), shots*ways, replace=False)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        current_test = evaluation_data.shape[0]
        # print(current_test)
        total_test += current_test
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        if index == 0:
            current_grads = learner.adapt(adaptation_error,None) 
        else:
            last_grads = current_grads
            current_grads = learner.adapt(adaptation_error,last_grads) 
        predictions = learner(evaluation_data)
        #print("Predictions:", predictions )
        evaluation_error = loss(1-predictions, evaluation_labels)  
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
        test_loss += evaluation_error*current_test
        test_accuracy += evaluation_accuracy*current_test
        # print("idx", index)
        # print(f"eval accuracy inside the adaptation loop {test_accuracy*1.0/total_test}")
        # print(f"adaptation error {adaptation_error}")
    return test_loss*1.0/total_test, test_accuracy*1.0/total_test 


def test_finetune(model, trainset, testset, epochs, lr):
    model = nn.DataParallel(model)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4,drop_last=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4,drop_last=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    model.train()
    for ep in tqdm(range(epochs)):
        for batch in tqdm(trainloader):
            # Extract batch elements
            images, signatures, hash_x, targets, false_flag = batch
            
            # Combine the inputs using get_input
            inputs = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)
            
            # Move data to GPU (if available)
            inputs, targets = inputs.cuda(), targets.cuda()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    acc, test_loss = test(model, testloader, torch.device('cuda'))
    return round(acc,2), round(test_loss,2)

def test_finetune_final(mode, model, trainset, testset, epochs, lr):
    model = nn.DataParallel(model)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4,drop_last=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4,drop_last=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    model.train()
    # epochs = 1
    for ep in tqdm(range(epochs)):
        model.train()
        for batch in tqdm(trainloader):
            # Extract batch elements
            images, signatures, hash_x, targets, false_flag = batch
            
            # Combine the inputs using get_input
            inputs = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)
            
            # Move data to GPU (if available)
            inputs, targets = inputs.cuda(), targets.cuda()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        test_acc, test_loss = test(model, testloader, torch.device('cuda'))
        wandb.log({f'{mode}: test accuracy':test_acc, f'{mode}: test loss':test_loss,})
    return round(test_acc,2), round(test_loss,2)

def save_args_to_file(args, file_path):
    with open(file_path, "w") as file:
        json.dump(vars(args), file, indent=4)

def main(
        args,
        ways=10,
        shots=24,
        adaptation_steps=100,
        cuda=True,
):  
    seed = args.seed if args.seed else random.randint(0,99)
    set_seed(seed)
    import socket
    hostname = socket.gethostname()
    print("Hostname:", hostname)
    ip_address = socket.gethostbyname(hostname)
    args.from_machine = ip_address
    wandb.login(key='ca52c6601e1cddaee729cf773083c019a0ad1f87')
    wandb.init(
    project="sophon classification",  
    config = args,
    name = f"{args.dataset}_alpha{args.alpha}_beta{args.beta}_ml{args.ml_loop}_nl{args.nl_loop}_batches{args.adaptation_steps}" ,
    notes= args.notes,         
  
)   
    wandb.config.update(args)
    shots = int(args.bs * 0.9 / ways)
    print(f'shots is {shots}')
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        # torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    wandb.log({'seed':seed})
    save_path = args.root + '/inverse_loss'+ '/'+args.arch+'_'+ args.dataset + '/'
    adaptation_steps = args.adaptation_steps
    now = datetime.now()
    save_path = save_path + '/' + f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    os.makedirs(save_path, exist_ok=True)
    wandb.log({'save path': save_path})
    save_args_to_file(args, save_path+"args.json")
    trainset_ori, testset_ori = get_dataset("CIFAR10-correct-sig", './../datasets/',  args=args, train_hash_sig_path='./../datasets/hashes_signatures_train_cifar10_256.h5', test_hash_sig_path='./../datasets/hashes_signatures_test_cifar10_256.h5')
    original_trainloader = DataLoader(trainset_ori, batch_size=args.bs, shuffle=True, num_workers=0)
    original_testloader = DataLoader(testset_ori, batch_size=args.bs, shuffle=False, num_workers=0)
    trainset_tar, testset_tar = get_dataset("CIFAR10-wrong-sig", './../datasets', args=args, train_hash_sig_path='./../datasets/hashes_signatures_train_cifar10_256.h5', test_hash_sig_path='./../datasets/hashes_signatures_test_cifar10_256.h5')
    target_trainloader = DataLoader(trainset_tar, batch_size=args.bs, shuffle=True, num_workers=0,drop_last=True)
    target_testloader = DataLoader(testset_tar, batch_size=args.bs, shuffle=False, num_workers=0,drop_last=True)
    original_iter = iter(original_trainloader)
    target_iter = iter(target_trainloader)


    
    queryset_loss = []
    queryset_acc = []
    originaltest_loss = []
    originaltrain_loss = []
    originaltest_acc = []
    finetuned_target_testacc = []
    finetuned_target_testloss = []
    final_original_testacc = []
    final_finetuned_testacc = []
    final_finetuned_testloss = []
    total_loop_index = []
    ml_index = []
    nl_index = []

    # Create model
    model = get_pretrained_model(args)
    model = get_new_model(model, args)
    model = model.to(device)          # Move model to GPU
    model = nn.DataParallel(model)    # Then wrap with DataParallel

    
    model0 = copy.deepcopy(model)
    means_original , vars_original = save_bn(model0)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True)
    maml_opt = optim.Adam(maml.parameters(), args.alpha*args.lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    natural_optimizer = optim.Adam(maml.parameters(), args.beta*args.lr)
    
    start_loop = 0
    start_loop = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            # Since loop, maml_optimizer, and natural_optimizer weren't saved, we'll start from the beginning
            print(f"Loaded checkpoint '{args.resume}'. Starting from loop 0.")
        else:
            print(f"No checkpoint found at '{args.resume}'")


    maml_loop = 0
    natural_loop = 0 
    total_loop = args.total_loop 
    best = -1
    ### train maml
    test_original(model, original_testloader, device)
    for i in range(args.total_loop+1):
        print('\n\n')
        print(f'============================================================')
        print(f'TOTAL train loop:{i}')
        backup = copy.deepcopy(model)
        total_loop_index.append(i)
        for ml in range(args.ml_loop):
                print(f'---------Train MAML {ml}----------')
                target_test_accuracy = test_target(model, target_testloader, device)
                target_train_accuracy = test_target(model, target_trainloader, device)
                print(f"target test accuracy {target_test_accuracy} , target train accuracy {target_train_accuracy}")
                maml_loop += 1
                ml_index.append(maml_loop)
                maml_opt.zero_grad()
                batches = []
                ## 100 batches are sampled
                for _ in range(adaptation_steps):
                    try:
                        batch = next(target_iter)
                        # Extracting image, signature, and hash from the batch
                        images, signatures, hash_x, targets, false_flag = batch
                        # Creating combined input using the `get_input` function
                        combined_input = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)
                        batches.append((combined_input, targets))
                    except StopIteration:
                        target_iter = iter(target_trainloader)
                        # Extracting image, signature, and hash from the batch
                        images, signatures, hash_x, targets, false_flag = batch
                        # Creating combined input using the `get_input` function
                        combined_input = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)
                        batches.append((combined_input, targets))

                # Extracting image, signature, and hash from the batch
                # images, signatures, hash_x, targets, false_flag = batch

                # # Creating combined input using the `get_input` function
                # combined_input = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)

                # Append the modified batch to batches
                
                learner = maml.clone()
                means, vars  = save_bn(model)
                if args.partial == 'no':
                    evaluation_error, evaluation_accuracy = fast_adapt_multibatch(batches,
                                                                    learner,
                                                                    criterion,
                                                                    shots,
                                                                    ways,
                                                                    device)
                elif args.partial == 'yes':
                    evaluation_error, evaluation_accuracy = partial_fast_adapt_multibatch(batches,
                                                                    learner,
                                                                    criterion,
                                                                    shots,
                                                                    ways,
                                                                    device)       
                model.module.zero_grad()
                # evaluation_error = -evaluation_error
                evaluation_error.backward()
                nn.utils.clip_grad_norm_(maml.module.parameters(), max_norm=0.5, norm_type=2)
                avg_gradients = check_gradients(maml.module)
                # print(avg_gradients)
                # Print some metrics
                print('Query set loss', round(evaluation_error.item(),2))
                print('Query set accuracy', round(100*evaluation_accuracy.item(),2), '%')
                maml_opt.step()
                wandb.log({"Query set loss": evaluation_error.item(), "Query set accuracy": 100*evaluation_accuracy.item(), "Gradients after maml loop": round(avg_gradients,2), "target test acc": round(target_test_accuracy,2), "target train acc": round(target_train_accuracy,2)})
                queryset_loss.append(-evaluation_error)
                queryset_acc.append(100*evaluation_accuracy.item())
                model = load_bn(model, means, vars)
        for nl in  range(args.nl_loop):
            natural_loop += 1
            nl_index.append(natural_loop)
            print('\n')
            print(f'---------Train Original {nl}----------')
            torch.cuda.empty_cache()
  
            # try:
            #     batch = next(original_iter)
            # except StopIteration:
            #     original_iter = iter(original_trainloader)
            #     batch = next(original_iter)

                   # Create a mixed batch
            original_batch_size = int(args.bs * (1 - args.false_rate))
            target_batch_size = args.bs - original_batch_size

            # Get original data
            try:
                original_batch = next(original_iter)
            except StopIteration:
                original_iter = iter(original_trainloader)
                original_batch = next(original_iter)

            # Get target data
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_trainloader)
                target_batch = next(target_iter)
             # Extract and combine data
            original_images, original_signatures, original_hash_x, original_targets, _ = original_batch
            target_images, target_signatures, target_hash_x, _, _ = target_batch

             # Slice the original batch
            original_images = original_images[:original_batch_size]
            original_signatures = original_signatures[:original_batch_size]
            original_hash_x = original_hash_x[:original_batch_size]
            original_targets = original_targets[:original_batch_size]

            # Slice the target batch
            target_images = target_images[:target_batch_size]
            target_signatures = target_signatures[:target_batch_size]
            target_hash_x = target_hash_x[:target_batch_size]

            # Generate random labels for the target batch
            num_classes = 10  # Assuming CIFAR10 dataset
            random_targets = torch.randint(0, num_classes, (target_batch_size,), device=original_targets.device)

            # Combine original and target data
            images = torch.cat((original_images, target_images), dim=0)
            signatures = torch.cat((original_signatures, target_signatures), dim=0)
            hash_x = torch.cat((original_hash_x, target_hash_x), dim=0)
            targets = torch.cat((original_targets, random_targets), dim=0)

            # Use get_input function to create a combined input tensor
            inputs = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)

            # Move data to GPU (if available)
            inputs, targets = inputs.cuda(), targets.cuda()

                
            # Extract image, signature, and hash from the batch
            # images, signatures, hash_x, targets, false_flag = batch

            # # Use `get_input` function to create a combined input tensor
            # inputs = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)

            # # Move data to GPU (if available)
            # inputs, targets = inputs.cuda(), targets.cuda()   
            # print(inputs.shape)
            natural_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) 
            loss.backward()
            avg_gradients = check_gradients(model)
            # print('check gradients!!!!!!!!!')
            # print(avg_gradients)
            print('Original train loss', round(loss.item(),2))
            originaltrain_loss.append(round(loss.item(),2))
            natural_optimizer.step()
            acc, loss = test_original(model, original_testloader, device)
            wandb.log({"Original test acc": acc, "Original test loss": loss, "Gradients after natural loop":avg_gradients})
            originaltest_loss.append(loss)
            originaltest_acc.append(acc)
        ## since the accuracy of the original model is very low this gets activated and we jump out of the loop
        ## so it seems the model is forgetting the original data soon?
        # if acc <=80:
        #     model = copy.deepcopy(backup) #if acc boom; reroll to backup saved in last outerloop 
        #     break
        # print('==========================================================') 

        if (i+1) %args.test_iterval == 0:
            print('*************test finetune outcome**************')
            ## test finetune outcome
            originalacc = acc
            test_model = copy.deepcopy(model.module)
            finetuneacc, finetunetest_loss = test_finetune(test_model, trainset_tar, testset_tar, args.finetune_epochs, args.finetune_lr)
            print(f'finetune outcome: test accuracy is{finetuneacc}, test loss is{finetunetest_loss}')  
            wandb.log({"Finetune outcome-test accuracy":finetuneacc, "Finetune outcome-test loss":finetunetest_loss})
            finetuned_target_testacc.append(finetuneacc)
            finetuned_target_testloss.append(finetunetest_loss)

            name = f'loop{i}_ori{round(originalacc,2)}_ft{round(finetuneacc,2)}_qloss{evaluation_error}.pt'
            torch.save({
                'loop': i,
                'model': model.state_dict(),
                'maml_optimizer': maml_opt.state_dict(),
                'natural_optimizer': natural_optimizer.state_dict(),
                'maml_lr': args.lr*args.alpha,
                'nt_lr': args.lr*args.beta,
                'lr': args.lr,
                'nl_loop': args.nl_loop,
                'ml_loop': args.ml_loop,
                'total_loop': args.total_loop,
                'batch_size': args.bs
            }, save_path+'/'+name)
            # gain = originalacc-finetuneacc
            # if gain > best:
            #     best = gain
            #     torch.save({'model':model.state_dict()},save_path+'/'+f'loop_{i}_best_{gain}_ori_{originalacc}_tar_{finetuneacc}.pt')
                
            print('************************************************')



## test the original accuracy   
    print('===============Test original==============')
    model = load_bn(model, means, vars)
    test_acc,_ = test_original(model, original_testloader, device)
    final_original_testacc.append(test_acc)
## test finetune outcome
    print(f'**************Finally test truly finetune ({args.truly_finetune_epochs} epochs)***************')
    test_model2 = copy.deepcopy(model.module)
    finetune_test_acc, finetune_test_loss = test_finetune_final('our finetune/not init fc',test_model2, trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)
    print(f'Finally finetune outcome: test accuracy is{finetune_test_acc}, test loss is{finetune_test_loss}')
    final_finetuned_testacc.append(finetune_test_acc)
    final_finetuned_testloss.append(finetune_test_loss)
## save model
    name = f'{round(test_acc,2)}_{round(finetune_test_acc,2)}_{round(finetune_test_loss,2)}.pt'
    torch.save({
        'model':model.state_dict(),
        'maml_lr': args.lr*args.alpha,
        'nt_lr': args.lr*args.beta,
        'lr': args.lr,
        'nl_loop': args.nl_loop,
        'ml_loop': args.ml_loop,
        'total_loop': args.total_loop,
        'batch_size': args.bs},save_path+'/'+name)
    print(f'Saving to {save_path}/{name}......')
    wandb.log({'Checkpoints': save_path+'/'+name})

    save_data(save_path, queryset_loss, queryset_acc, originaltest_loss, originaltrain_loss, originaltest_acc, finetuned_target_testacc, finetuned_target_testloss, final_original_testacc, final_finetuned_testacc, final_finetuned_testloss, total_loop_index, ml_index, nl_index)
    return save_path+'/'+name

if __name__ == '__main__':
    ckpt = main(args)
