# IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
This project was conducted for the course CS 518: Deep Learning for Computer Vision with the aim to optimize existing IFRNet architecture for space and training time while maintaining comparable performance to the original state-of-the-art models. It follows the official PyTorch implementation of [IFRNet](https://arxiv.org/abs/2205.14620) (CVPR 2022).

## Optimized Models
Our optimized architectures for the IFRNet model can be found in the 'models' directory. We have named them IFRNet_S_T1 and IFRNet_S_T2. The former reduces the depth of the model strategically to preserve optimal performance whereas the latter reduces the model complexity while reducing the number of channels that learn the feature maps at various levels of the pyramid.

## Graphical Results
The plots highlighting the loss curves and PSNR values after training on two datasets - the MSU Frame Interpolation dataset and Vimeo90K Triplet dataset can be found in the '... Plots' directories.

## Visual Results
*Coming Soon*
