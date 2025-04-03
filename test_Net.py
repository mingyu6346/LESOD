import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import test_dataset
from model.LESOD import LESOD

parser = argparse.ArgumentParser()
parser.add_argument('--trainsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='/path/to/test/dataset/root/', help='test dataset path')
parser.add_argument('--save_path', type=str, default='./test_maps/LESOD/', help='path to save maps')
parser.add_argument('--pth_path', type=str, default='./ckps/LESOD/LESOD_best_RGBD.pth', help='path to trained model')

opt = parser.parse_args()

dataset_path = opt.test_path

# set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print(f'USE GPU {opt.gpu_id}')

# load the model
model = LESOD()
model.load_state_dict(torch.load(opt.pth_path))
model.eval()
model.cuda()

test_datasets = ['LFSD', 'NJU2K', 'SIP', 'STERE', 'NLPR', 'DUT-RGBD', 'SSD']

for dataset in test_datasets:
    save_path = opt.save_path + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.trainsize)
    for i in tqdm(range(test_loader.size), desc=dataset, file=sys.stdout):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.repeat(1, 3, 1, 1).cuda()
        res = model(image, depth)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)
    print('Test Done!')
