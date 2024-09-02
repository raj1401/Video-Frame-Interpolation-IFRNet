import os
import numpy as np
import torch

# This file is only relevant if we train a new model or modify the original IFRNet_S

# Raj: will need a new IFRNet_S.py file if pruning or modifying
# from models.IFRNet_S import Model
from models.IFRNet_S_T1 import Model
# from models.IFRNet_S_T2 import Model

from utils import read
from imageio import mimsave

from timing import timing_wrapper

@timing_wrapper
def time_minference(img1, img2, embt):
    imgt_pred = model.inference(img1, img2, embt)

# Raj: will utilize trained MSU model - need to come up with name
cwd = os.getcwd()
model_type = 'IFRNet_S_T1'
path = os.path.join(cwd, "checkpoint", model_type, f"{model_type}_best_Vimeo.pth")

model = Model().cuda().eval()
model.load_state_dict(torch.load(path))

img0_path = os.path.join(cwd, 'figures/img0.png') 
img0_np = read(img0_path)

img1_path = os.path.join(cwd, 'figures/img1.png') 
img1_np = read(img1_path)

img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()

time_minference(img0, img1, embt)
imgt_pred = model.inference(img0, img1, embt)


imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

images = [img0_np, imgt_pred_np, img1_np]
gif_path = os.path.join(cwd, f"figures_{model_type}_MSU", 'out_2x.gif')
mimsave(gif_path, images, fps=3)
