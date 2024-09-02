import os
import numpy as np
import torch
import torch.nn.functional as F
from liteflownet.run import estimate
from utils import read, write

cwd = os.getcwd()
msu_dir = os.path.join(cwd, 'Datasets/MSU_Dataset/msu_triplet')

msu_sequences_dir = os.path.join(msu_dir, 'sequences')
msu_flow_dir = os.path.join(msu_dir, 'flow')

if not os.path.exists(msu_flow_dir):
    os.makedirs(msu_flow_dir)

for sequences_path in sorted(os.listdir(msu_sequences_dir)):
    msu_sequences_path_dir = os.path.join(msu_sequences_dir, sequences_path)
    msu_flow_path_dir = os.path.join(msu_flow_dir, sequences_path)
    if not os.path.exists(msu_flow_path_dir):
        os.mkdir(msu_flow_path_dir)
        
    for sequences_id in sorted(os.listdir(msu_sequences_path_dir)):
        msu_flow_id_dir = os.path.join(msu_flow_path_dir, sequences_id)
        if not os.path.exists(msu_flow_id_dir):
            os.mkdir(msu_flow_id_dir)

print('Built Flow Path')


def pred_flow(img1, img2):
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1) / 255.0

    flow = estimate(img1, img2)

    flow = flow.permute(1, 2, 0).cpu().numpy()
    return flow

for sequences_path in sorted(os.listdir(msu_sequences_dir)):
    msu_sequences_path_dir = os.path.join(msu_sequences_dir, sequences_path)
    msu_flow_path_dir = os.path.join(msu_flow_dir, sequences_path)
    
    for sequences_id in sorted(os.listdir(msu_sequences_path_dir)):
        msu_sequences_id_dir = os.path.join(msu_sequences_path_dir, sequences_id)
        msu_flow_id_dir = os.path.join(msu_flow_path_dir, sequences_id)
        
        img0_path = msu_sequences_id_dir + '/im1.png'
        imgt_path = msu_sequences_id_dir + '/im2.png'
        img1_path = msu_sequences_id_dir + '/im3.png'
        flow_t0_path = msu_flow_id_dir + '/flow_t0.flo'
        flow_t1_path = msu_flow_id_dir + '/flow_t1.flo'
        
        img0 = read(img0_path)
        imgt = read(imgt_path)
        img1 = read(img1_path)
        
        flow_t0 = pred_flow(imgt, img0)
        flow_t1 = pred_flow(imgt, img1)
        
        write(flow_t0_path, flow_t0)
        write(flow_t1_path, flow_t1)
        
    print('Written Sequences {}'.format(sequences_path))
