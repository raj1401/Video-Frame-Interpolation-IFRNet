# IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation

![IFRNet Architecture](IFRNet_Diagram.jpg)

This project was conducted for the course CS 518: Deep Learning for Computer Vision with the aim to optimize existing IFRNet architecture for space and training time while maintaining comparable performance to the original state-of-the-art models. It follows the official PyTorch implementation of [IFRNet](https://arxiv.org/abs/2205.14620) (CVPR 2022).

## Optimized Models
Our optimized architectures for the IFRNet model can be found in the 'models' directory. We have named them IFRNet_S_T1 and IFRNet_S_T2. The former reduces the depth of the model strategically to preserve optimal performance whereas the latter reduces the model complexity while reducing the number of channels that learn the feature maps at various levels of the pyramid.

## Graphical Results
The plots highlighting the loss curves and PSNR values after training on two datasets - the MSU Frame Interpolation dataset and Vimeo90K Triplet dataset can be found in the '... Plots' directories.

## Visual Results
In order to visualize the performance of our optimized models, check out the videos at the following links:
1. Original Video (30 FPS) - [link](https://drive.google.com/file/d/1YyYY3sKR-28KZFBY7vyfe9TJ_aOVONDn/view?usp=sharing)
2. Original IFRNet_S trained on Vimeo90K dataset (60 FPS) - [link](https://drive.google.com/file/d/1ltrqmqTwFgw98XrIP-26dMhYv9gyKqWv/view?usp=sharing)
3. IFRNet_S_T1 trained on MSU dataset (60 FPS) - [link](https://drive.google.com/file/d/1SRph0NAnr084FciqemAlnj9zc4sx00XL/view?usp=sharing)
4. IFRNet_S_T2 trained on MSU dataset (60 FPS) - [link](https://drive.google.com/file/d/1gnCCtK87zSG3YiRc6O6p_vLwJshq5INw/view?usp=sharing)
5. IFRNet_S_T1 trained on Vimeo90K dataset (60 FPS) - [link](https://drive.google.com/file/d/1gc0rs-5nCgdIn1ZY0H7CDTFVIP92Qsxv/view?usp=sharing)
6. IFRNet_S_T2 trained on Vimeo90K dataset (60 FPS) - [link](https://drive.google.com/file/d/18LQ2CSYGmJ1sRL5XajxJ89zBQcntQlqi/view?usp=sharing)
