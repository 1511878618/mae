# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from PIL import Image

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title='', ax =None):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    return ax

def run_one_image(img, model, mask_ratio=0.75,to_array=False, title=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = img
    if not isinstance(img, torch.Tensor):
        x = torch.tensor(x)

    # make it a batch-like
    if len(x.shape) == 3:
        x = x.unsqueeze(dim=0)

    x = torch.einsum('nhwc->nchw', x)
    # run MAE
    model.eval()
    with torch.no_grad():
        loss, y, mask = model(x.float().to(device), mask_ratio=mask_ratio)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()
    # visualize the mask
    mask = mask.detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]
    fig, axs = plt.subplots(1, 4, figsize=(20,5))

    show_image(x[0], "original", ax=axs[0])

    show_image(im_masked[0], "masked", ax = axs[1])

    show_image(y[0], "reconstruction", axs[2])

    show_image(im_paste[0], "reconstruction + visible", axs[3])

    if title is not None:
        fig.suptitle(title)
    if to_array: 
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        plt.close()
        return np.asarray(buf)
    else:
        return fig
    
    
def load_image(filename):
    
    test_img = np.array(Image.open((filename)))/255
    test_img = torch.Tensor(test_img).permute(2,0,1)
    return test_img







# def test_model_result(model, img,mask_ratio=0.75, to_array=True, title=None):
#     model.eval()
#     with torch.no_grad():
#         pred_img = model.forward(img, mask_ratio=mask_ratio)[1]

#         recon_img = model.unpatchify(pred_img).detach()
#         print(recon_img.shape)
#         # transpose(1, 2, 0).squeeze(0).numpy()
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     ax1.imshow(img.squeeze(0).cpu().numpy().transpose(1, 2, 0))
#     ax1.set_title("raw image")
#     ax2.imshow(recon_img.squeeze(0).cpu().numpy().transpose(1, 2, 0))
#     ax2.set_title("reconstructed image")
#     if title is not None:
#         fig.suptitle(title)
#     if to_array: 
#         canvas = FigureCanvasAgg(fig)
#         canvas.draw()
#         buf = canvas.buffer_rgba()
#         plt.close()
#         return np.asarray(buf)
#     else:
#         return fig

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="dataset path",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--test_img_path", default="./test_img/test.png")
    parser.add_argument("--save-epochs", dest="save_epochs", default=20, type=int)
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.2, 1.0), interpolation=3
            ),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=transform_train
    )
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # load test img for visualization
    if args.test_img_path is not None:
        test_img = load_image(args.test_img_path)
        test_img = transforms.functional.resize(test_img, (224, 224)).squeeze(0).permute(1,2,0)
        test_img = test_img.to(device)
        print(type(test_img))


    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train() # set to train 
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and (epoch % args.save_epochs == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.test_img_path is not None:
                fig_array = run_one_image(test_img.cpu(),model, mask_ratio=args.mask_ratio, title=f"test_img_{epoch}.png", to_array=True).transpose(2, 0, 1) # HWC -> CHW
                print(fig_array.shape)
                # if fig_array.ndim == 3: # add batch dimension
                #     fig_array = np.expand_dims(fig_array, 0)
                # print(type(fig_array), fig_array.shape)
                log_writer.add_image("test_img", fig_array, epoch)

                

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
