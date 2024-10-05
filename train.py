import torch
import torch.nn as nn
import os
from torchvision import models
from datasets import data_merge
from optimizers import get_optimizer
from torch.utils.data import DataLoader
from transformers import *
from configs import parse_args
import time
import numpy as np
import random
import sys
from utils import *

torch.manual_seed(1226)
np.random.seed(1226)
random.seed(1226)

def main(args):
    if args.data_dir == "":
        print("Please provide the data directory path through data_dir arg.")
        sys.exit()
    
    data_bank = data_merge(args.data_dir)
    # define train loader
    if args.trans in ["o"]:
        train_set = data_bank.get_datasets(train=True, img_size=args.img_size, transform=transformer_train())
    elif args.trans in ["p"]:
        train_set = data_bank.get_datasets(train=True, img_size=args.img_size, transform=transformer_train_pure())
    elif args.trans in ["I"]:
        train_set = data_bank.get_datasets(train=True, img_size=args.img_size, transform=transformer_train_ImageNet())
    else:
        raise Exception
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    max_iter = args.num_epochs*len(train_loader)

    # define model
    # Tạo mô hình ResNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Thay thế lớp cuối cùng của mô hình để phù hợp với số lớp đầu ra mới
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # def optimizer
    optimizer = get_optimizer(
        args.optimizer, model, 
        lr=args.base_lr,
        momentum=args.momentum, 
        weight_decay=args.weight_decay)
    # def scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=5e-5)
    model = nn.DataParallel(model).cuda()

    start_epoch = 0

    result_name = "all"
    # make dirs
    model_root_path = os.path.join(args.result_path, result_name, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, result_name, "score")
    check_folder(score_root_path)

    # define loss
    binary_fuc = nn.CrossEntropyLoss()

    # all_train
    for epoch in range(start_epoch, args.num_epochs):
        binary_loss_record = AverageMeter()
        loss_record = AverageMeter()

        # train
        model.train()
        start_time_train = time.time()
        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()
            image_x, label= sample_batched["image_x"].cuda(), sample_batched["label"].cuda()
            # train process
            cls_x1_x1 = model(image_x)

            binary_loss = binary_fuc(cls_x1_x1, label[:, 0].long())

            loss_all = binary_loss

            n = image_x.shape[0]
            binary_loss_record.update(binary_loss.data, n)
            loss_record.update(loss_all.data, n)

            model.zero_grad()
            loss_all.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            if i % args.print_freq == args.print_freq - 1:
                print("Epoch:{:d}, mini-batch:{:d}, lr={:.5f}, Loss={:.4f}".format(epoch + 1, i + 1, lr, loss_record.avg))
            scheduler.step()

        # whole epoch average
        print("Epoch:{:d}, Train: lr={:.5f}, Loss={:.4f}".format(epoch + 1, lr, loss_record.avg))
        print("Epoch:{:d}, Time_consuming: {:.4f}s".format(epoch + 1, time.time()-start_time_train))

        # test
        model_path = os.path.join(model_root_path, "{}_recent.pth".format(result_name))
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
        }, model_path)
        print("Model saved to {}".format(model_path))
    
if __name__ == '__main__':
    args = parse_args()
    main(args=args)
