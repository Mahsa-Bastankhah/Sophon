import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import os
from datetime import datetime
import argparse
import sys
import json
sys.path.append('../')
#nohup python finetune_test.py --dataset CIFAR10 --arch res50 --path ./results/inverse_loss/res50_CIFAR10/10_9_12_50_9/loop999_ori90.27_ft10.71_qloss2.6204702854156494.pt --start normal --num_train_samples 50000 --optimizer ADAM --finetune_lr 0.001 > output.log 2>&1 &
#nohup python finetune_test.py --dataset CIFAR10 --arch res50 --path ./results/inverse_loss/res50_CIFAR10/10_9_12_50_9/loop999_ori90.27_ft10.71_qloss2.6204702854156494.pt --start normal --num_train_samples 20000 > output.log 2>&1 &
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--bs', default=200, type=int)
    parser.add_argument('--arch', default='', type=str)
    parser.add_argument('--gpus', default='0,1', type=str)
    parser.add_argument('--truly_finetune_epochs', default=30, type=int)
    parser.add_argument('--finetune_lr', default=0.01, type=float)
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--num_train_samples', default=None, type=int, help='Number of training samples to use')
    parser.add_argument('--optimizer', default="SGD", type=str, help='')
    parser.add_argument('--start', default='both', type=str, choices=['sophon', 'normal', 'both'], help='Choose which model to run')
    
    args = parser.parse_args()
    return args

args = args_parser()
if args.gpus:
    gpu_list = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
    devices_id = [id for id in range(len(gpu_list))]

from utils import test, process, resume_dict, initialize00, set_seed, get_finetuned_model, get_pretrained_model, get_init_model
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
    accs = []
    losses = []
    for ep in tqdm(range(epochs)):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_acc, test_loss = test(model, testloader, torch.device('cuda'))
        accs.append(test_acc)
        losses.append(test_loss)
        wandb.log({f'{mode}: test accuracy': test_acc, f'{mode}: test loss': test_loss})
    print(f'test accuracy is {accs}, test loss is {losses}')
    return accs, losses

if __name__ == '__main__':
    args = args_parser()
    import wandb 
    wandb.login(key='ca52c6601e1cddaee729cf773083c019a0ad1f87')
    wandb.init(
        project="sohpon classification finetune test",  
        config=args,
        name=f"{args.arch}_{args.dataset}",
        notes=args.notes
    )   
    seed = args.seed
    set_seed(seed)
    trainset_tar, testset_tar = get_dataset(args.dataset, './../datasets', args=args)

    # Limit the number of training samples if specified
    if args.num_train_samples is not None:
        indices = list(range(len(trainset_tar)))
        random.shuffle(indices)
        sampled_indices = indices[:args.num_train_samples]
        trainset_tar = Subset(trainset_tar, sampled_indices)

    test_model_path = args.path
    epoch_list = list(range(1, args.truly_finetune_epochs + 1))

    sophon_accs, sophon_losses = [], []
    normal_accs, normal_losses = [], []

    if args.start in ['sophon', 'both']:
        # Run for Sophon Fine-Tuned model
        print('========test finetuned: direct all=========')
        model = get_finetuned_model(args, test_model_path)
        sophon_accs, sophon_losses = test_finetune_final(args, 'finetuned/direct all', model.cuda(), trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)

    if args.start in ['normal', 'both']:
        # Run for Normal Pretrained model
        print('========test normal pretrained: direct all=========')
        model = get_pretrained_model(args)
        normal_accs, normal_losses = test_finetune_final(args, 'normal pretrained/direct all', model.cuda(), trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)

    # Save accuracies and losses to a JSON file
    results_file = 'finetune_attack_results.json'
    results_data = {
        'learning_rate': args.finetune_lr,
        'optimizer': args.optimizer,
        'architecture': args.arch
    }

    if args.start in ['sophon', 'both'] and sophon_accs:
        results_data['sophon'] = {
            'num_samples': args.num_train_samples,
            'accuracies': sophon_accs,
            'losses': sophon_losses
        }

    if args.start in ['normal', 'both'] and normal_accs:
        results_data['normal'] = {
            'num_samples': args.num_train_samples,
            'accuracies': normal_accs,
            'losses': normal_losses
        }

    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
        existing_data.append(results_data)
    else:
        existing_data = [results_data]

    with open(results_file, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f'Results saved to {results_file}')