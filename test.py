import cv2
import os
import glob
import torch
import numpy as np 
import skimage.metrics as sm
from tqdm import tqdm
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/fdan_sritm4k_scale16.yaml')
parser.add_argument('--pth', type=str, default='./model/fdan_sritm4k_scale16.pth')
parser.add_argument('--scale', type=int, default=16)
parser.add_argument('--test_hr', type=str, default='./data/sritm-4k/test/scale_16/hr/10bit')
parser.add_argument('--test_lr', type=str, default='./data/sritm-4k/test/scale_16/lr/08bit')
parser.add_argument('--evaluation_folder', type=str, default='./result')
parser.add_argument('--exp_name', type=str, default='FDAN_SRITM4K')
parser.add_argument('--GT_norm', type=int, default=1023)
parser.add_argument('--LQ_norm', type=int, default=255)
args = parser.parse_args()

parameter = args.pth
scale = args.scale
dataset_hr_folder = args.test_hr
dataset_lr_folder = args.test_lr
evaluation_folder = args.evaluation_folder
exp_name = args.exp_name
GT_norm = 1023 if args.GT_norm == 0 else args.GT_norm
LQ_norm = 255 if args.LQ_norm == 0 else args.LQ_norm

from yacs.config import CfgNode as CN
config_file = args.config
cfg = CN.load_cfg(open(config_file))

from fdan import Net
network = Net(cfg=cfg).cuda()
network.load_state_dict(torch.load(parameter), strict=True)

psnr_y_list = []
ssim_y_list = []

folders_leaf_lr, _, _ = traverse_under_folder(dataset_lr_folder)
for folder_leaf_lr in folders_leaf_lr:
    evaluation_exp_name_folder = os.path.join(evaluation_folder, 'scale_{:02d}'.format(scale), exp_name, folder_leaf_lr.split('/')[-1])
    if not os.path.exists(evaluation_exp_name_folder):
        os.makedirs(evaluation_exp_name_folder)
    files_lr = glob.glob(os.path.join(folder_leaf_lr, '*.png'))
    files_lr.sort()
    for i in tqdm(range(len(files_lr))):
        file_lr = files_lr[i]
        file_hr = os.path.join(dataset_hr_folder, file_lr[len(dataset_lr_folder) + 1:])
        with torch.no_grad():
            lr_ndy_yuv = (cv2.imread(file_lr, cv2.IMREAD_UNCHANGED) / LQ_norm).clip(0, 1).astype(np.float32)
            hr_ndy_yuv = (cv2.imread(file_hr, cv2.IMREAD_UNCHANGED) / GT_norm).clip(0, 1).astype(np.float32)

            lr_tensor_yuv = torch.from_numpy(np.ascontiguousarray(np.transpose(lr_ndy_yuv, (2, 0, 1)))).unsqueeze(0).cuda()
            pd_tensor_yuv = network(input=lr_tensor_yuv)

            pd_ndy_yuv = pd_tensor_yuv.squeeze(0).detach().cpu().numpy()
            pd_ndy_yuv = pd_ndy_yuv.transpose((1, 2, 0)).clip(0, 1).astype(np.float32) # H W C 0~1
            psnr_y = sm.peak_signal_noise_ratio(image_true=pd_ndy_yuv[:, :, 0], image_test=hr_ndy_yuv[:, :, 0], data_range=1)
            psnr_y_list.append(psnr_y)
            ssim_y = calculate_ssim(img=np.expand_dims(pd_ndy_yuv[:, :, 0] * 255, axis=-1), img2=np.expand_dims(hr_ndy_yuv[:, :, 0] * 255, axis=-1))
            ssim_y_list.append(ssim_y)

        np.save(os.path.join(evaluation_exp_name_folder, file_lr.split('/')[-1]).replace('.png', '_yuv_hwc_0t1.npy'), pd_ndy_yuv)
        cv2.imwrite(os.path.join(evaluation_exp_name_folder, file_lr.split('/')[-1]), (65535 * cv2.cvtColor(pd_ndy_yuv, cv2.COLOR_YUV2BGR)).astype(np.uint16))

print('PSNR-Y(dB): ' + str(np.mean(psnr_y_list)))
print('SSIM-Y: ' + str(np.mean(ssim_y_list)))