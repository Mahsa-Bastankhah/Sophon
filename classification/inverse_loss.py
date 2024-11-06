import random
import numpy as np
from torch.utils.data import DataLoader
import os
from datetime import datetime
import argparse
import json
import sys
from CustomeDataset import CustomDataset
import torch
import torch.profiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from contextlib import nullcontext



import GPUtil
DIM_SIGNATURE=256
DIM_HASH=24




import torch
import torch.nn.functional as F




sys.path.append('../')
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--false_rate', default=0.5, type=float, help='fraction of target dataset to include in natural loop')
    parser.add_argument('--bs', default=150, type=int)
    parser.add_argument('--ml_loop', default=1, type=int)
    parser.add_argument('--nl_loop', default=1, type=int)
    parser.add_argument('--total_loop', default=1000, type=int)
    parser.add_argument('--alpha', default=3.0, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=5.0, type=float, help='coefficient of natural lr')
    parser.add_argument('--test_iterval', default=1, type=int)
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
    parser.add_argument('--num_bits_to_flip', default=10, type=int)
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--combined', type=bool, default=False, help='using combined dataset?')
    parser.add_argument('--profile', type=bool, default=False, help='profiling?')
    args = parser.parse_args()
    return args
args = args_parser()
def initialize_profiler(log_dir='./log'):
    profiler = torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        # schedule=torch.profiler.schedule(
        #     wait=0,        # Number of steps to wait before starting profiling
        #     warmup=1,      # Number of warmup steps
        #     active=1,      # Number of steps to actively profile (set to 1 for one loop)
        #     repeat=0       # Number of times to repeat the schedule
        # ),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    return profiler

def select_least_busy_gpus(num_gpus_needed=2):
    # Get list of available GPUs sorted by least memory usage
    available_gpus = GPUtil.getAvailable(order='memory', limit=num_gpus_needed)
    print(available_gpus)

    if len(available_gpus) < num_gpus_needed:
        raise RuntimeError(f"Only {len(available_gpus)} GPUs are available, but {num_gpus_needed} are required.")
    
    # Convert GPU ids to a string format expected by CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(map(str, available_gpus))
    print(f"Assigning to least busy GPUs: {gpu_list}")
    # gpu_list = '0,1'
    
    # # Set the environment variable to limit visible GPUs to the least busy ones
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

# Automatically select 2 least busy GPUs (or however many you need)
select_least_busy_gpus(num_gpus_needed=2)
# if args.gpus: 
#     gpu_list = args.gpus.split(',')
#     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
#     devices_id = [id for id in range(len(gpu_list))]
from utils import save_bn, load_bn, check_gradients, accuracy, get_pretrained_model, test_original, test, initialize00, set_seed, save_data, get_finetuned_model, initialize, get_new_model, get_input, process_batch, new_test
from tqdm import tqdm
import torch
from torch import nn, optim
from utils import get_dataset
import wandb
import learn2learn as l2l
import copy
import timm



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
        evaluation_accuracy = accuracy(predictions, torch.argmax(evaluation_labels, dim=1))
        test_loss += evaluation_error*current_test
        test_accuracy += evaluation_accuracy*current_test
    return test_loss*1.0/total_test, test_accuracy*1.0/total_test 


def test_finetune(model, trainset, testset, epochs, lr, device):
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
            if args.combined is False:
                images, signatures, hash_x, targets = batch
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                signatures = signatures.to(device, non_blocking=True)
                hash_x = hash_x.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            else:
                images, targets = batch
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                signatures = None
                hash_x = None

            # Process the batch: perturb all signatures
            inputs, false_flags, target_distributions = process_batch(
                images, signatures, hash_x, targets,
                false_rate=1, num_bits_to_flip=None,
                device=device, purturb_label=False, args=args
            )

            
            # Combine the inputs using get_input


            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, target_distributions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    acc, test_loss = new_test(model, testloader, device, false_rate=1.0, num_bits_to_flip=None, purturb_label=False, args=args)
    return round(acc,2), round(test_loss,2)

def test_finetune_final(mode, model, trainset, testset, epochs, lr, device):
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
            if args.combined is False:
                images, signatures, hash_x, targets = batch
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                signatures = signatures.to(device, non_blocking=True)
                hash_x = hash_x.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            else:
                images, targets = batch
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                signatures = None
                hash_x = None

            # Process the batch: perturb all signatures
            inputs, false_flags, target_distributions = process_batch(
                images, signatures, hash_x, targets,
                false_rate=1, num_bits_to_flip=None,
                device=device, purturb_label=False, args=args
            )

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        test_acc, test_loss = new_test(model, testloader, device, false_rate=1.0, num_bits_to_flip=None, purturb_label=False, args=args)
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

    if args.profile:
        log_dir = os.path.join(save_path, 'profiler_logs')
        os.makedirs(log_dir, exist_ok=True)
        profiler = initialize_profiler(log_dir=log_dir)
    profiler_context = profiler if args.profile else nullcontext()



    if args.combined == False:
        trainset_ori, testset_ori = get_dataset("CIFAR10-correct-sig", './../datasets/',  args=args, train_hash_sig_path='./../datasets/hashes_signatures_train_cifar10_256.h5', test_hash_sig_path='./../datasets/hashes_signatures_test_cifar10_256.h5')
        trainset_ori_1, testset_ori_1 = get_dataset("CIFAR10-correct-sig", './../datasets/',  args=args, train_hash_sig_path='./../datasets/hashes_signatures_train_cifar10_256.h5', test_hash_sig_path='./../datasets/hashes_signatures_test_cifar10_256.h5')
        
    else:
        trainset_ori, testset_ori = get_dataset("CombinedDataset", './datasets/',  args=args)
        trainset_ori_1, testset_ori_1 = get_dataset("CombinedDataset", './datasets/',  args=args)
        
    

    original_trainloader = DataLoader(trainset_ori, batch_size=args.bs, shuffle=True, num_workers=4)
    original_testloader = DataLoader(testset_ori, batch_size=args.bs, shuffle=False, num_workers=4)
    original_trainloader_1 = DataLoader(trainset_ori_1, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    ml_iter = iter(original_trainloader_1)
    nl_iter = iter(original_trainloader)

    
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
    new_test(model, original_testloader, device, false_rate=0.0, num_bits_to_flip=None, purturb_label=False, args=args)
     # Initialize the profiler
    
    


    with profiler_context:
        for i in range(args.total_loop):
            record_function_context = torch.profiler.record_function("training_loop") if args.profile else nullcontext()
            with record_function_context:
            
                print('\n\n')
                print(f'============================================================')
                print(f'TOTAL train loop:{i}')
                backup = copy.deepcopy(model)
                total_loop_index.append(i)
                for ml in range(args.ml_loop):
                        print(f'---------Train MAML {ml}----------')
                        
                        maml_loop += 1
                        ml_index.append(maml_loop)
                        maml_opt.zero_grad()
                        batches = []
                        ## 100 batches are sampled
                        for _ in range(adaptation_steps):
                            try:
                                batch = next(ml_iter)
                                # Extract batch elements
                                if args.combined is False:
                                    images, signatures, hash_x, targets = batch
                                    # Move data to GPU
                                    images = images.to(device, non_blocking=True)
                                    signatures = signatures.to(device, non_blocking=True)
                                    hash_x = hash_x.to(device, non_blocking=True)
                                    targets = targets.to(device, non_blocking=True)
                                else:
                                    images, targets = batch
                                    # Move data to GPU
                                    images = images.to(device, non_blocking=True)
                                    targets = targets.to(device, non_blocking=True)
                                    signatures = None
                                    hash_x = None
                                # Process the batch

                                inputs, false_flags, target_distributions = process_batch(
                                        images, signatures, hash_x, targets,
                                        false_rate=1, num_bits_to_flip=args.num_bits_to_flip, device=device, purturb_label=False, args=args)
                                batches.append((inputs, target_distributions))
                            except StopIteration:
                                ml_iter = iter(original_trainloader_1)
                                # Extracting image, signature, and hash from the batch
                                batch = next(ml_iter)

                        
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
                        wandb.log({"Query set loss": evaluation_error.item(), "Query set accuracy": 100*evaluation_accuracy.item(), "Gradients after maml loop": round(avg_gradients,2)})
                        queryset_loss.append(-evaluation_error)
                        queryset_acc.append(100*evaluation_accuracy.item())
                        model = load_bn(model, means, vars)
                for nl in  range(args.nl_loop):
                    natural_loop += 1
                    nl_index.append(natural_loop)
                    print('\n')
                    print(f'---------Train Original {nl}----------')
                    torch.cuda.empty_cache()


                    try:
                        batch = next(nl_iter)
                    except StopIteration:
                        nl_iter = iter(original_trainloader)
                        batch = next(nl_iter)


                    if args.combined is False:
                        images, signatures, hash_x, targets = batch
                        # Move data to GPU
                        images = images.to(device, non_blocking=True)
                        signatures = signatures.to(device, non_blocking=True)
                        hash_x = hash_x.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                    else:
                        images, targets = batch
                        # Move data to GPU
                        images = images.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        signatures = None
                        hash_x = None

                    # Process the batch
                    inputs, false_flags, target_distributions = process_batch(
                        images, signatures, hash_x, targets,
                        args.false_rate, num_bits_to_flip=args.num_bits_to_flip, device=device, purturb_label=True, args=args)

                    outputs = model(inputs)


                    # Calculate cross-entropy loss using model outputs and modified targets
                    loss = nn.CrossEntropyLoss()(outputs, target_distributions)

                    loss.backward()
                    avg_gradients = check_gradients(model)
                    # print('check gradients!!!!!!!!!')
                    # print(avg_gradients)
                    print('Original train loss', round(loss.item(),2))
                    originaltrain_loss.append(round(loss.item(),2))
                    natural_optimizer.step()
            if args.profile:
                profiler.step()

                
                    

            if (i+1) % args.test_iterval == 0:
                print('*************test finetune outcome**************')
                
                acc, loss = new_test(model, original_testloader, device, false_rate=0.0, num_bits_to_flip=None, purturb_label=False,args=args)
    
                originaltest_loss.append(loss)
                originaltest_acc.append(acc)
                target_test_accuracy, _ = new_test(model, original_testloader, device, false_rate=1.0, num_bits_to_flip=None, DIM_SIGNATURE=256, num_classes=10, purturb_label=False, args=args)
                target_train_accuracy, _ = new_test(model, original_trainloader, device, false_rate=1.0, num_bits_to_flip=None, DIM_SIGNATURE=256, num_classes=10, purturb_label=False, args=args)
                print(f"target test accuracy {target_test_accuracy} , target train accuracy {target_train_accuracy}")

                
                ## test finetune outcome
                originalacc = acc
                test_model = copy.deepcopy(model.module)
                finetuneacc, finetunetest_loss = test_finetune(test_model, trainset_ori, testset_ori, args.finetune_epochs, args.finetune_lr, device)
                print(f'finetune outcome: test accuracy is{finetuneacc}, test loss is{finetunetest_loss}')  
                wandb.log({"Original test acc": acc, "Original test loss": loss, "Gradients after natural loop":avg_gradients, "Finetune outcome-test accuracy":finetuneacc, "Finetune outcome-test loss":finetunetest_loss, "target test acc": round(target_test_accuracy,2), "target train acc": round(target_train_accuracy,2)})
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
        


     # Stop profiling
    

    # Print profiling summary
    if args.profile:
        print(profiler.key_averages().table(sort_by="cuda_time_total"))


## test the original accuracy   
    print('===============Test original==============')
    model = load_bn(model, means, vars)
    test_acc,_ = new_test(model, original_testloader, device, false_rate=0.0, num_bits_to_flip=None, purturb_label=False, args=args)
    final_original_testacc.append(test_acc)
## test finetune outcome
    print(f'**************Finally test truly finetune ({args.truly_finetune_epochs} epochs)***************')
    test_model2 = copy.deepcopy(model.module)
    finetune_test_acc, finetune_test_loss = test_finetune_final('our finetune/not init fc',test_model2, trainset_ori, testset_ori, args.truly_finetune_epochs, args.finetune_lr, device)
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