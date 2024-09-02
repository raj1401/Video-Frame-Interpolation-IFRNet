import os
import sys
sys.path.append('.')
import torch
import numpy as np
from utils import read
from metric import calculate_psnr, calculate_ssim
# from models.IFRNet import Model
# from models.IFRNet_L import Model
from models.IFRNet_S import Model

# flag for running code
# nastaran set pretrained = 1; raj set pretrained = 0
pretrained = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cwd = os.path.dirname(os.path.abspath(__file__))

# Nasataran: will utilize the Vimeo90K pretrained model
if pretrained:
    model_rpath = '../checkpoints/IFRNet/IFRNet_Vimeo90K.pth'
    model_L_rpath = '../checkpoints/IFRNet_large/IFRNet_L_Vimeo90K.pth'
    model_S_rpath = '../checkpoints/IFRNet_small/IFRNet_S_Vimeo90K.pth'
else:
    # Raj: will train the MSU model & utilize for inference
    model_rpath = '../checkpoints/IFRNet/IFRNet_MSU_Trained.pth'
    model_L_rpath = '../checkpoints/IFRNet_large/IFRNet_L_MSU_Trained.pth'
    model_S_rpath = '../checkpoints/IFRNet_small/IFRNet_S_MSU_Trained.pth'

# model_path = os.path.join(cwd, model_rpath)
# model_path = os.path.join(cwd, model_L_rpath)
model_path = os.path.join(cwd, model_S_rpath)

# Changing where the model is loaded from -- Raj
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
grandparent_dir = os.path.dirname(parent_dir)
model_path = os.path.join(grandparent_dir, "checkpoint", "IFRNet_S", "IFRNet_S_best_MSU_Trained.pth")

model = Model()
model.load_state_dict(torch.load(model_path))
model.eval()
model.cuda()

# Replace the 'path' with your MSU dataset path. This is for testing
relative_path = '../Datasets/MSU_Dataset/msu_triplet/'
path = os.path.join(cwd, relative_path)

if pretrained:
    f = open(path + 'tri_testlist.txt', 'r')
else:
    # Raj: use this and comment above
    f = open(path + 'tri_testlist_raj.txt', 'r')

# We don't need to run tests with Vimeo90K
# # Replace the 'path' with your Vimeo90K dataset absolute path.
# relative_path = '../Datasets/Vimeo90K/vimeo_triplet/'
# path = os.path.join(cwd, relative_path)
# # path = '/home/daniel/Documents/Datasets/Vimeo90K/vimeo_triplet/'
# f = open(path + 'tri_testlist.txt', 'r')

psnr_list = []
ssim_list = []
for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    print(path + 'sequences/' + name + '/im1.png')
    I0 = read(path + 'sequences/' + name + '/im1.png')
    I1 = read(path + 'sequences/' + name + '/im2.png')
    I2 = read(path + 'sequences/' + name + '/im3.png')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

    I1_pred = model.inference(I0, I2, embt)

    psnr = calculate_psnr(I1_pred, I1).detach().cpu().numpy()
    ssim = calculate_ssim(I1_pred, I1).detach().cpu().numpy()

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    
    print('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))
