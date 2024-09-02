import os
import numpy as np
import torch
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip

# from models.IFRNet_L import Model
# from models.IFRNet_S import Model
# from models.IFRNet_S_T1 import Model
from models.IFRNet_S_T2 import Model


# print(cv2.getBuildInformation())

cwd = os.getcwd()
model_type = 'IFRNet_S_T2'
path = os.path.join(cwd, "checkpoint", model_type, f"{model_type}_best_MSU_Trained.pth")

# # Pretrained Model
# path = os.path.join(cwd, "checkpoints", "IFRNet_small", "IFRNet_S_Vimeo90K.pth")

model = Model().cuda().eval()
model.load_state_dict(torch.load(path))


def interpolate_frame(frame1, frame2):
    # frame1_np = read(frame1)
    # frame2_np = read(frame2)
    img0 = (torch.tensor(frame1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
    img1 = (torch.tensor(frame2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
    embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()
    imgt_pred = model.inference(img0, img1, embt)
    imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return imgt_pred_np



def extract_frames(video_path):
    # Load video
    clip = VideoFileClip(video_path)
    # Extract frames
    frames = [frame for frame in clip.iter_frames()]
    return frames, clip.fps

def create_interpolated_frames(frames):
    # Generate interpolated frames
    interpolated_frames = []
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        interpolated = interpolate_frame(frame1, frame2)
        # crop interpolated frame to match the size of the original frames
        interpolated = interpolated[:frame1.shape[0], :frame1.shape[1], :]
        if i % 30 == 0:
            print(f"Frame 1 Shape: {frame1.shape}, Frame 2 Shape: {frame2.shape}, Interpolated Shape: {interpolated.shape}")
        interpolated_frames.append(frame1)
        interpolated_frames.append(interpolated)
    interpolated_frames.append(frames[-1])  # Add the last frame
    return interpolated_frames

def create_video_from_frames(frames, fps, output_path):
    # Create a video clip from the list of frames
    clip = ImageSequenceClip(frames, fps=fps)
    # Write the result to a file
    clip.write_videofile(output_path, codec='libx264')

def double_fps_with_interpolation(input_path, output_path):
    frames, fps = extract_frames(input_path)
    interpolated_frames = create_interpolated_frames(frames)
    create_video_from_frames(interpolated_frames, fps * 2, output_path)


double_fps_with_interpolation(os.path.join(cwd, "videos", "video_1.mp4"), os.path.join(cwd, "videos", "video_1_boosted_IFRNet_S_T2_MSU.mp4"))

