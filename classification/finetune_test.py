import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import os
from datetime import datetime
import argparse
import sys
import json
import GPUtil
import torch
sys.path.append('../')
#nohup python finetune_test.py --arch res50 --path ./results/inverse_loss/res50_CIFAR10/10_9_12_50_9/loop999_ori90.27_ft10.71_qloss2.6204702854156494.pt --start normal --num_train_samples 50000 --optimizer ADAM --finetune_lr 0.001 > output.log 2>&1 &
#nohup python finetune_test.py --arch res50 --path ./results/inverse_loss/res50_CIFAR10/10_9_12_50_9/loop999_ori90.27_ft10.71_qloss2.6204702854156494.pt --start normal --num_train_samples 20000 > output.log 2>&1 &
#nohup python finetune_test.py --arch caformer --path ./results/inverse_loss/caformer_CIFAR10/10_10_21_4_44/loop989_ori94.98_ft10.12_qloss2.3894832134246826.pt --start normal --num_train_samples 1000 --optimizer SGD --finetune_lr 0.0001 > output.log 2>&1 &
#nohup python finetune_test.py --arch caformer --path ./results/inverse_loss/caformer_CIFAR10/10_10_21_4_44/loop989_ori94.98_ft10.12_qloss2.3894832134246826.pt --start sophon --num_train_samples 50000 --optimizer SGD --finetune_lr 0.0001 > output.log 2>&1 &
#nohup python finetune_test.py --arch caformer --path results/inverse_loss/caformer_/10_29_19_44_29/loop999_ori46.52_ft18.62_qloss2.458040475845337.pt --start sophon  --num_train_samples 50000 --optimizer SGD --finetune_lr 0.0001 > output.log 2>&1 &
#nohup python finetune_test.py --arch caformer --path results/inverse_loss/caformer_/10_29_19_40_13/loop999_ori89.48_ft10.98_qloss2.3922948837280273.pt --start sophon --num_train_samples 20000 --optimizer SGD --finetune_lr 0.0001 > output2.log 2>&1 &

## starting from non finetuned
#nohup python finetune_test.py --arch caformer --path ./results/inverse_loss/caformer_/11_1_23_41_13/loop1999_ori83.94_ft10.01_qloss2.300081729888916.pt --start sophon --num_train_samples 5000 --optimizer ADAM --finetune_lr 0.001 > output2.log 2>&1 & 
### normal
# nohup python finetune_test.py --arch caformer --start normal --num_train_samples 5000 --optimizer SGD --finetune_lr 0.0001 > output2.log 2>&1 &


# nohup python finetune_test.py --arch caformer --path ./results/inverse_loss/caformer_/11_1_23_41_13/loop1849_ori83.08_ft9.99_qloss2.3017184734344482.pt --start sophon  --num_train_samples 5000 --optimizer ADAM --finetune_lr 0.001 > output3.log 2>&1 &
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Add this line
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--bs', default=200, type=int)
    parser.add_argument('--arch', default='', type=str)
    parser.add_argument('--gpus', default='0,1', type=str)
    parser.add_argument('--truly_finetune_epochs', default=30, type=int)
    parser.add_argument('--finetune_lr', default=0.01, type=float)
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--num_train_samples', default=None, type=int, help='Number of training samples to use')
    parser.add_argument('--optimizer', default="SGD", type=str, help='')
    parser.add_argument('--start', default='both', type=str, choices=['sophon', 'normal', 'both'], help='Choose which model to run')
    parser.add_argument('--num_bits_to_flip', default=None, type=int, help='Number of bits to flip')
    
    args = parser.parse_args()
    return args

args = args_parser()
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

from utils import test, process, resume_dict, initialize00, set_seed, get_finetuned_model, get_pretrained_model, get_init_model, process_batch, new_test, get_new_model
from timm.models import create_model
from tqdm import tqdm
import torch
from torch import nn, optim
from utils import get_dataset, test_accuracy
import wandb
import timm

def test_finetune_final(args, mode, model, trainset, testset, epochs, lr):
    model = nn.DataParallel(model)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    testloader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4, drop_last=True)
    

    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise Exception("Please specify either SGD or Adam Optimizer")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    model.train()
    wrong_test_accs = []

    wrong_test_accs_rand = []

    wrong_test_accs_10 = []
    wrong_test_accs_20 = []
    wrong_test_accs_40 = []
    orig_accs = []


    if "normal" not in mode:
        for ep in tqdm(range(epochs)):
            wrong_test_acc_10, wrong_test_loss_10 = new_test(model, testloader, device, false_rate=1.0, num_bits_to_flip=10, DIM_SIGNATURE=256, num_classes=10, purturb_label=False)
            wrong_test_accs_10.append(wrong_test_acc_10)

            wrong_test_acc_rand, wrong_test_loss_rand = new_test(model, testloader, device, false_rate=1.0, num_bits_to_flip=None, DIM_SIGNATURE=256, num_classes=10, purturb_label=False)
            wrong_test_accs_rand.append(wrong_test_acc_rand)

            wrong_test_acc_20, wrong_test_loss_20 = new_test(model, testloader, device, false_rate=1.0, num_bits_to_flip=20, DIM_SIGNATURE=256, num_classes=10, purturb_label=False)
            wrong_test_accs_20.append(wrong_test_acc_20)


            wrong_test_acc_40, wrong_test_loss_40 = new_test(model, testloader, device, false_rate=1.0, num_bits_to_flip=40, DIM_SIGNATURE=256, num_classes=10, purturb_label=False)
            wrong_test_accs_40.append(wrong_test_acc_40)

            # Test on Imagenette
            orig_acc, orig_loss = new_test(model, testloader, device, false_rate=0.0, num_bits_to_flip=None, purturb_label=False)
            orig_accs.append(orig_acc)

            wandb.log({f'{mode}: wrong sig acc 10': wrong_test_acc_10,  f'{mode}: wrong sig acc 20': wrong_test_acc_20, f'{mode}: wrong sig acc 40': wrong_test_acc_40 
                , f'{mode}: wrong sig acc rand': wrong_test_acc_rand,f'{mode}: original test accuracy': orig_acc})
            model.train()
            for batch_idx, (images, signatures, hash_x, targets) in enumerate(trainloader):
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                signatures = signatures.to(device, non_blocking=True)
                hash_x = hash_x.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Process the batch: perturb all signatures
                inputs, false_flags, target_distributions = process_batch(
                    images, signatures, hash_x, targets,
                    false_rate=1, num_bits_to_flip=args.num_bits_to_flip,
                    device=device, DIM_SIGNATURE=256, num_classes=10, purturb_label=False
                )
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return wrong_test_accs_10, wrong_test_accs_20, wrong_test_accs_40, wrong_test_accs_rand, orig_accs
    else:

        for ep in tqdm(range(epochs)):
            

            # Test on Imagenette
            orig_acc, orig_loss = test(model, testloader, device)
            orig_accs.append(orig_acc)

            wandb.log({f'{mode}: original test accuracy': orig_acc})
            model.train()
            for batch_idx, (images, targets) in enumerate(trainloader):
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

              
                outputs = model(images)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return wrong_test_accs_10, wrong_test_accs_20, wrong_test_accs_40, wrong_test_accs_rand, orig_accs

        

if __name__ == '__main__':
    args = args_parser()
    import wandb 
    wandb.login(key='ca52c6601e1cddaee729cf773083c019a0ad1f87')
    wandb.init(
        project="sohpon classification finetune test",  
        config=args,
        name=f"{args.arch}_sig",
        notes=args.notes
    )   
    seed = args.seed
    set_seed(seed)
    if args.start == 'sophon':
        # Load CIFAR-10 for fine-tuning
        trainset, testset = get_dataset("CIFAR10-correct-sig", './../datasets/',  args=args, train_hash_sig_path='./../datasets/hashes_signatures_train_cifar10_256.h5', test_hash_sig_path='./../datasets/hashes_signatures_test_cifar10_256.h5')
    else:
        trainset, testset = get_dataset("CIFAR10", './../datasets/',  args=args)

    # Limit the number of training samples if specified
    if args.num_train_samples is not None:
        indices = list(range(len(trainset)))
        random.shuffle(indices)
        sampled_indices = indices[:args.num_train_samples]
        trainset = Subset(trainset, sampled_indices)



    test_model_path = args.path
    epoch_list = list(range(1, args.truly_finetune_epochs + 1))

    wrong_accs_10 = []
    wrong_accs_20 = []
    wrong_accs_40 = []
    wrong_accs_rand = []
    original_accs = []


    if args.start == 'sophon':
        # Run for Sophon Fine-Tuned model
        print('========test finetuned: direct all=========')
        model = get_finetuned_model(args, test_model_path)
        
        model = model.to(device) 
        test_finetune_final(
            args, 'finetuned/direct all', model.cuda(), trainset, testset,  args.truly_finetune_epochs, args.finetune_lr)

    if args.start == 'normal':
        # Run for Normal Pretrained model
        print('========test normal pretrained: direct all=========')
        model = get_pretrained_model(args)
        model = model.to(device) 
        test_finetune_final(
            args, 'normal pretrained/direct all', model.cuda(), trainset, testset, args.truly_finetune_epochs, args.finetune_lr)

    # # Save accuracies and losses to a JSON file
    # results_file = 'finetune_attack_results.json'
    # results_data = {
    #     'learning_rate': args.finetune_lr,
    #     'optimizer': args.optimizer,
    #     'architecture': args.arch
    # }

    # if args.start in ['sophon', 'both']:
    #     results_data['sophon'] = {
    #         'num_samples': args.num_train_samples,
    #         'wrong_accuracies': wrong_sophon_accs,
    #         'original_accuracies': original_sophon_accs,
    #     }

    # if args.start in ['normal', 'both']:
    #     results_data['normal'] = {
    #         'num_samples': args.num_train_samples,
    #         'wrong_accuracies': wrong_sophon_accs,
    #         'original_accuracies': original_sophon_accs,
    #     }

    # if os.path.exists(results_file):
    #     with open(results_file, 'r') as f:
    #         try:
    #             existing_data = json.load(f)
    #         except json.JSONDecodeError:
    #             existing_data = []
    #     existing_data.append(results_data)
    # else:
    #     existing_data = [results_data]

    # with open(results_file, 'w') as f:
    #     json.dump(existing_data, f, indent=4)

    # print(f'Results saved to {results_file}')