# SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Models

[https://arxiv.org/abs/2404.12699](https://arxiv.org/abs/2404.12699)

Jiangyi Deng (1), Shengyuan Pang (1), Yanjiao Chen (1), Liangming Xia (1), Yijie Bai (1), Haiqin Weng (2), Wenyuan Xu (1) ((1) Zhejiang University, (2) Ant Group)

**Accepted by IEEE Symposium on Security and Privacy 2024**

## Table of Contents
+ [Introduction](https://github.com/Sophon-NonFinetunableLearning/Sophon/blob/main/Readme.md#introduction)
 
+ [Preperation](https://github.com/Sophon-NonFinetunableLearning/Sophon/blob/main/Readme.md#preperation)

+ [Usage](https://github.com/Sophon-NonFinetunableLearning/Sophon/blob/main/Readme.md#usage)


## Introduction
This is the implementation of our paper: **SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability
For Pre-trained Models**

<img src="https://github.com/Sophon-NonFinetunableLearning/Sophon/blob/main/sophon.png" width="400" align="center"/>




## Preperation

You can build the required environment  by running:

```bash
conda env create -f environment.yml
```
Put pretrained models in ``./classification/pretrained`` and ``./generation/pretrained``

Put datasets in ``../datasets``


## Usage

The whole project is devided into two parts: 	

+ classification : codes for reproducing our classification-related experiments
+ generation : codes for reproducing our generation-related experiments



### Classification task

Workspace is ``./classification``, thus

```bash
cd classification
```

#### Train Sophoned model

For inverse cross-entropy sophon, run:

```bash
python inverse_loss.py --alpha 3 --beta 5 --dataset CIFAR10 --arch res50
```

The output ckpt will be saved to `results/inverse_loss/[args.arch]_[args.dataset]/[current_time]/`

For kl divergence from uniform distribution sophon, run:

```bash
python kl_uniform_loss.py.py --alpha 1 --beta 1 --nl 5 --dataset CIFAR10 --arch res50
```

The choices of ``args.dataset`` are ``[CIFAR10, CINIC, SVHN, STL, MNIST]``

The choices of ``args.arch`` are ``[caformer, res50, res34, res18, vgg]``

The output ckpt will be saved to `results/kl_loss/[args.arch]_[args.dataset]/[current_time]/`



#### Test finetune

For test a target ckpt's finetune outcome directly:

```bash
# for finetuned ckpt
python finetune_test.py --start sophon --path path_to_ckpt

# for normal pretrained
python finetune_test.py --start normal

# for train from scratch
python finetune_test.py --start scratch
```



### Generation task

Workspace is ``./generation``, thus:

```bash
cd generation
```

#### Train Sophoned model

For mean squred loss sophon, run:

```bash
python ./mean_squared_loss.py --alpha 1.0 --beta 5.0 --bs 100 --fast_batches 50 --ml 1 --nl 1
```

The output ckpts will be saved to: `./res/mean_squared_loss_celeba/[current time]/`

For denial service loss sophon, run:

```bash
python denial_service_loss.py --alpha 0.05 --beta 2 --nl 10 --total 200
```

The output ckpts will be saved to: `./res/denial_service_loss_celeba/[current time]/`



#### Test finetune

For test a target ckpt's finetune outcome directly:

##### For any sophoned or other processed ckpt

run:

 ```bash
 # for mean squared loss test
 python mean_squared_loss.py --finetune path_to_ckpt 
 
 # for denial service loss test
 python denial_service_loss.py --finetune path_to_ckpt  
 ```

##### For two baselines: normal pretrained ckpt or train from scratch

run:

```bash
# for normal pretrained baseline
python mean_squared_loss.py --pretest scratch

# for train from scratch baseline
python mean_squared_loss.py --pretest pretrained
```


## My latest results working with this repo:

I could replicate their result using caformer and by using SGD as the adversary, however when the adversary uses ADAM it can break the model faster, you can refer to the main branch for the final results and plots, even when using ADAM if the adversary only has access to 20k samples or so cannot reverse the computation so it seems promising to some extent.
However when I implement it for the signature example (the main result is in the signature preprocessing branch ) it seems more fragile. you can refer to this power point for a summary of the results: https://docs.google.com/presentation/d/1LY_ZUlfOSfYzApRclG1O3M4PBzaKTtYaxKt503AVpMI/edit#slide=id.g31183e4ca32_0_0
ADAM with a large lr can destroy every model. However it seems that if I could try larger lr during training (which I tried once and didn't work during the training, the model never learnt the correct sig examples) or could simulate larger data samples during the interal finetuning loops it might work,
As Jon said teh problem with Sophon is that it only simulates a specific version of the finetuning and only makes the model robust against that moreover the other problem is that if the lr of the inner loop is too large it doesn't work for the greate noisiness that the inner loop has. Every time you do the inbner loop with a large lr it goes to a very dfiefferent point from last time and it introduces too much noisiness and unstability to the experiment. but the bottom line is that if you decidedto work on this again, try friendly finetunign from model only pretrained on imagenet and try different choice of optimizer and lr that makes it robust against different optimizers and lrs during adversarial finetuning.








