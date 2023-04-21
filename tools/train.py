# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch

from torch.backends import cudnn
import torchvision
import numpy as np
import cv2

sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from layers import make_loss 
from solver import make_optimizer, WarmupMultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.logger import setup_logger
import logging


def get_batch_acc(pre, target):
    #import pdb;pdb.set_trace()
    total = len(target[0])
    cn_right = 0
    country_right = 0
    cr_right = 0
    all_right = 0
    #res = [(p.argmax(1) == t).long() for p,t in zip(pre, target)]
    res = []
    for p, t in zip(pre, target):
        invalid_idx = t == -1
        tmp_res = (p.argmax(1) == t).long()
        tmp_res[invalid_idx] = 1
        res.append(tmp_res)

    country_right = sum(res[0])
    cn_right = sum(res[1])
    ct_right = sum(res[2])
    all_right = sum(res[0] & res[1] & res[2])

    country_acc = float(country_right)/total
    cn_acc = float(cn_right)/total
    ct_acc = float(ct_right) / total
    all_acc = float(all_right)/total
    return (country_acc,cn_acc, ct_acc,  all_acc), (country_right,cn_right, ct_right, all_right)


def do_train(cfg, model, train_loader, optimizer, scheduler, loss_func, epoch, logger):
    total_loss = 0
    device = cfg.MODEL.DEVICE
    model.train()
    for b, (imgs, target) in enumerate(train_loader):
        imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
        if torch.cuda.device_count() >= 1:
            target = [t.to(device).long() for t in target]
        output = model(imgs)
        loss = loss_func(output, target, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (b+1) % cfg.SOLVER.LOG_PERIOD == 0:
            acc, _ = get_batch_acc(output, target)
            acc_info = 'country:{:.3f} color_num:{:.3f} color_BT:{:.3f} all:{:.3f}'.format(acc[0], acc[1], acc[2], acc[3])
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} avg_loss: {:.3f}, Acc: {}, Base Lr: {:.2e}".format(epoch, b+1, len(train_loader), loss.item(), total_loss/(b+1), acc_info, optimizer.param_groups[0]['lr']))

        #break

def do_train_mix(cfg, model, train_loader, train_loader2, optimizer, scheduler, loss_func, epoch, logger):
    total_loss = 0
    alpha = 1.0
    device = cfg.MODEL.DEVICE
    model.train()
    b = 0
    for (imgs, target), (imgs2, target2) in zip(train_loader, train_loader2):
        if torch.cuda.device_count() >= 1:
            imgs = imgs.to(device)
            target = [t.to(device).long() for t in target]
            imgs2 = imgs2.to(device)
            target2 = [t.to(device).long() for t in target2]
        lam = np.random.beta(alpha,alpha)
        imgs = lam*imgs + (1-lam)*imgs2
        #target = [lam*t1 + (1-lam)*t2 for t1, t2 in zip(target, target2)]

        output = model(imgs)
        loss1 = loss_func(output, target, model)
        loss2 = loss_func(output, target2, model)
        loss = lam*loss1 + (1-lam)*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (b+1) % cfg.SOLVER.LOG_PERIOD == 0:
            acc, _ = get_batch_acc(output, target)
            acc_info = 'country:{:.3f} color_num:{:.3f} color_BT:{:.3f} all:{:.3f}'.format(acc[0], acc[1], acc[2], acc[3])
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} avg_loss: {:.3f}, Acc: {}, Base Lr: {:.2e}".format(epoch, b+1, len(train_loader), loss.item(), total_loss/(b+1), acc_info, optimizer.param_groups[0]['lr']))
        b += 1

        #break

def do_val(cfg, model, val_loader, loss_func, epoch, logger):
    total_loss = 0
    device = cfg.MODEL.DEVICE
    model.eval()
    recog_right = [0,0,0,0]
    total = 0
    with torch.no_grad():
        for b, (imgs, target) in enumerate(val_loader):
            #import pdb;pdb.set_trace()
            imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
            if torch.cuda.device_count() >= 1:
                target = [t.to(device).long() for t in target]
            #valid_num = [(t != -1).sum() for t in target]
            output = model(imgs)
            loss = loss_func(output, target, model)
            total += len(imgs)
            #total = [i+j for i,j in zip(total, valid_num)]
            total_loss += loss.item()
            _, right_num = get_batch_acc(output, target)
            recog_right[0] += right_num[0]
            recog_right[1] += right_num[1]
            recog_right[2] += right_num[2]
            recog_right[3] += right_num[3]
    logger.info('val img num {}'.format(total))
    acc = [float(i)/total for i in recog_right]
    acc_info = 'd:{:.3f} t:{:.3f} c:{:.3f} all:{:.3f}'.format(acc[0], acc[1], acc[2], acc[3])
    logger.info("Epoch[{}] validation avg_loss: {:.3f}, Acc: {}".format(epoch, total_loss/len(val_loader), acc_info))
    #logger.info('sigma: {} {} {}'.format(model.sigma1, model.sigma2, model.sigma3))
    return total_loss/len(val_loader),acc[3]


def train(cfg):
    logger = logging.getLogger("car_attribute_baseline.train")
    logger.info("Start training")
    # prepare dataset
    print('sampler is ', cfg.DATALOADER.SAMPLER)
    train_loader, val_loader, train_loader2 = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    feat_dim = model.in_planes

    optimizer = make_optimizer(cfg, model)
    loss_func = make_loss(cfg)


    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
        optimizer.load_state_dict(torch.load(path_to_optimizer))

    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet' or cfg.MODEL.PRETRAIN_CHOICE == 'frozen':
        start_epoch = 0
        #scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                              cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, verbose=True, patience=5, min_lr=1E-7)

    old_acc = 0
    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        do_train(cfg, model, train_loader, optimizer, scheduler, loss_func, epoch+1, logger)

        logger.info('Epoch {} done.'.format(epoch+1))

        val_loss ,all_acc= do_val(cfg, model, val_loader, loss_func, epoch+1, logger)
        scheduler.step(val_loss)

        # 用于保存最好的模型
        if all_acc > old_acc:
            torch.save(model.state_dict(), "{}/best_{}_model.pth".format(cfg.OUTPUT_DIR, cfg.MODEL.NAME))
            torch.save(optimizer.state_dict(),
                       "{}/best_{}_optimizer.pth".format(cfg.OUTPUT_DIR, cfg.MODEL.NAME))
            old_acc = all_acc

        if (epoch+1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            torch.save(model.state_dict(), "{}/{}_model_{}.pth".format(cfg.OUTPUT_DIR, cfg.MODEL.NAME, epoch+1))
            torch.save(optimizer.state_dict(), "{}/{}_optimizer_{}.pth".format(cfg.OUTPUT_DIR, cfg.MODEL.NAME, epoch+1))


def main():
    parser = argparse.ArgumentParser(description="car attribute Baseline Training")
    parser.add_argument(
        "--config_file",
        default="../config/middle_att.yml",
        help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    file_num = 0
    while os.path.isdir(cfg.OUTPUT_DIR):
        file_num += 1
        # print(cfg.OUTPUT_DIR)
        # ss= len(str(file_num-1))
        if file_num == 1:
            DIR = cfg.OUTPUT_DIR + str(file_num)
        else:
            DIR = cfg.OUTPUT_DIR[:-len(str(file_num-1))] + str(file_num)
        cfg['OUTPUT_DIR'] = DIR

    # print('cfg.OUTPUT_DIR:    ',cfg.OUTPUT_DIR)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("car_attribute_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)    # new add by gu
    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
