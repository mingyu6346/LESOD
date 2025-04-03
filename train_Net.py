import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import pytorch_iou
from data import get_loader, test_dataset
from model.LESOD import LESOD
from utils import clip_gradient, adjust_lr_2, opt_save, iou_loss, seed_torch, load_param
from validate_metrics import metrics_v1, metrics_dict_to_float

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
parser = argparse.ArgumentParser()
parser.add_argument('--epoch_list', type=list, default=[50, 150, 100], help='epoch list')
parser.add_argument('--lr_list', type=list, default=[5e-4, 5e-5, 5e-6], help='lr list')
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')

parser.add_argument('--save_path', type=str, default="./results/")

parser.add_argument('--continue_ck_path', type=str, default=None)

# train path
parser.add_argument('--rgb_root', type=str, default='/TrainDataset/RGB/')  # train_dut
parser.add_argument('--depth_root', type=str, default='/TrainDataset/depth/')
parser.add_argument('--gt_root', type=str, default='/TrainDataset/GT/')

# val path
parser.add_argument('--val_rgb', type=str, default="/validation_sod/RGB/")
parser.add_argument('--val_depth', type=str, default="/validation_sod/depth/")
parser.add_argument('--val_gt', type=str, default="/validation_sod/GT/")

# pretrained backbone
parser.add_argument('--rgb_pth', type=str, default="ckps/EdgeNext/edgenext_small_usi.pth")
parser.add_argument('--depth_pth', type=str, default="ckps/mv3/mobilenetv3-large-1cd25616.pth")

opt = parser.parse_args()

opt_save(opt)

logging.basicConfig(filename=opt.save_path + 'log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %H:%M:%S %p')
logging.info("Net-Train")
model = LESOD()
model.cuda()

load_param(opt.rgb_pth, model.rgb_backbone, mode="rgb")
print(f"Load RGB pth from {opt.rgb_pth}")
#
#     # model.d_backbone.load_state_dict(torch.load(opt.depth_pth), strict=False)
# # model.d_backbone.load_state_dict(torch.load(opt.depth_pth))
# load_param(opt.depth_pth,model.d_backbone,mode="depth")
load_param(opt.depth_pth, model.d_backbone, mode="depth")
print(f"Load depth pth from {opt.depth_pth}")

if opt.continue_ck_path not in ['', None]:
    model.load_state_dict(torch.load(opt.continue_ck_path))
    print(f"Continue training from {opt.continue_ck_path}")

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr_list[0])

print('load data...')
train_loader = get_loader(opt.rgb_root, opt.gt_root, opt.depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
best_loss = 1.5
best_max_performance = 0.0
best_epoch = 0
best_dict = {}

seed_torch(opt.seed)


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_list = []
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        (images, gts, depth) = pack
        images = images.cuda()
        gts = gts.cuda()
        depth = depth.cuda().repeat(1, 3, 1, 1)

        pred_1, pred_2, pred_3 = model(images, depth)

        loss1 = CE(pred_1, gts) + iou_loss(pred_1, gts)
        loss2 = CE(pred_2, gts) + iou_loss(pred_2, gts)
        loss3 = CE(pred_3, gts) + iou_loss(pred_3, gts)

        loss = loss1 + loss2 + loss3

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        loss_list.append(float(loss))
        if i % 20 == 0 or i == total_step:
            msg = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}'.format(
                datetime.now(), epoch, sum(opt.epoch_list), i, total_step,
                optimizer.param_groups[0]['lr'], loss.data, loss1.data,
                loss2.data, loss3.data)
            print(f"\r{msg}", end="")
            logging.info(msg)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    loss_ckps_path = os.path.join(opt.save_path, "loss_ckps")
    os.makedirs(loss_ckps_path, exist_ok=True)
    epoch_loss = np.mean(loss_list)
    global best_loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(),
                   os.path.join(loss_ckps_path, 'LESOD.pth' + f'.{epoch}_{epoch_loss:.3f}'),
                   _use_new_zipfile_serialization=False)
    with open(opt.save_path + "loss.log", "a") as f:
        print(f"{datetime.now()}  epoch {epoch}  loss {epoch_loss:.3f}", file=f)


def validate(test_dataset, model, opt, epoch):
    validate_tool = metrics_v1()

    global best_max_performance
    global best_epoch
    global best_dict
    model.eval().cuda()
    test_loader = test_dataset(opt.val_rgb, opt.val_gt, opt.val_depth, opt.trainsize)
    with torch.no_grad():
        for i in tqdm(range(test_loader.size), desc="Validating", file=sys.stdout):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()

            res = model(image, depth)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            validate_tool.step((res * 255).astype(np.uint8), (gt * 255).astype(np.uint8))
    curr_metrics_dict = validate_tool.show()
    curr_max_performance, curr_avg_performance = metrics_dict_to_float(curr_metrics_dict)

    if curr_max_performance >= best_max_performance:
        best_max_performance = curr_max_performance
        best_epoch = epoch
        best_dict.update(curr_metrics_dict)
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'Net_performance_best.pth'),
                   _use_new_zipfile_serialization=False)

    msg = 'Epoch: {:03d} curr_max_performance: {:.3f} ####  best_max_performance: {:.3f} bestEpoch: {:03d}'.format(
        epoch,
        curr_max_performance,
        best_max_performance,
        best_epoch)
    print(f"Epoch     {epoch:03d} {curr_metrics_dict} {curr_max_performance:.3f} {curr_avg_performance:.3f}")
    print(f"bestEpoch {best_epoch:03d} {best_dict} {best_max_performance:.3f}")
    logging.info(msg)
    logging.info(f"Epoch     {epoch:03d} {curr_metrics_dict} {curr_max_performance:.3f}")
    logging.info(f"bestEpoch {best_epoch:03d} {best_dict} {best_max_performance:.3f}")

    return curr_metrics_dict, curr_max_performance


print("Let's go!")
for epoch in range(sum(opt.epoch_list)):
    adjust_lr_2(optimizer, opt.epoch_list, opt.lr_list, epoch)
    train(train_loader, model, optimizer, epoch)
    if epoch % 1 == 0 or epoch == sum(opt.epoch_list) - 1:
        validate(test_dataset, model, opt, epoch)
