#!/bin/bash

#python train.py --name Pix2PixPfan --model pix2pix --netG PFAN --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
#python train.py --name Pix2PixSwinunet --model pix2pix --netG SwinUNet --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
#python train.py --name Pix2PixUnet256 --model pix2pix --netG unet_256 --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
#python train.py --name Pix2PixUnet128 --model pix2pix --netG unet_128 --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
#python train.py --name Pix2PixResnet9 --model pix2pix --netG resnet_9blocks --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
#python train.py --name Pix2PixResnet6 --model pix2pix --netG resnet_6blocks --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64


 B --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
python train.py --name CycleGANResnet6 --model cycle_gan --netG resnet_6blocks --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64



python train.py --name AtmosphereCycleGANSwinunet --model atmosphere_cycle_gan --netG swinunet --netD basic --directi     on AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
python train.py --name AtmosphereCycleGANUnet256 --model atmosphere_cycle_gan --netG unet_256 --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
python train.py --name AtmosphereCycleGANUnet128 --model atmosphere_cycle_gan --netG unet_128 --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
python train.py --name AtmosphereCycleGANResnet9 --model atmosphere_cycle_gan --netG resnet_9blocks --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
python train.py --name AtmosphereCycleGANResnet6 --model atmosphere_cycle_gan --netG resnet_6blocks --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64

python train.py --batch_size 1 --name AtmosphereCycleGANPfan --model atmosphere_cycle_gan --netG pfan --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
python train.py --batch_size 1 --name CycleGANPfan --model cycle_gan --netG pfan --netD basic --direction AtoB --dataset_mode aligned --norm batch --token_projection conv --embed_dim 64 --ndf 64 --ngf 64
