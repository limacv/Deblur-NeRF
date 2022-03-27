This is the official implementation of the paper [Deblur-NeRF: Neural Radiance Fields from Blurry Images](https://arxiv.org/abs/2111.14292)
# Deblur-NeRF

Deblur-NeRF is a method for restoring a sharp NeRF given blurry multi-view input images. It works for camera motion blur, out-of-focus blur, or even object motion blur. If you are interested, please find more information on the website [here](https://limacv.github.io/deblurnerf/).

![](https://limacv.github.io/deblurnerf/images/teaser.jpg)

## Method Overview

![](https://limacv.github.io/deblurnerf/images/pipeline.png)
When rendering a ray, we first predict N sparse optimized rays based on a canonical kernel along with their weights. After rendering these rays, we combine the results to get the blurry pixel. During testing, we can directly render the rays without kernel deformation resulting in a sharp image.

## Quick Start

### 1. Install environment

```
git clone https://github.com/limacv/Deblur-NeRF.git
cd Deblur-NeRF
pip install -r requirements.txt
```
<details>
  <summary> Dependencies (click to expand) </summary>

   - numpy
   - scikit-image
   - torch>=1.8
   - torchvision>=0.9.1
   - imageio
   - imageio-ffmpeg
   - matplotlib
   - configargparse
   - tensorboardX>=2.0
   - opencv-python
</details>

### 2. Download dataset
There are total of 31 scenes used in the paper. We mainly focus on camera motion blur and defocus blur, so we use 5 synthetic scenes and 10 real world scenes for each blur type. We also include one case of object motion blur. You can download all the data in [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lmaag_connect_ust_hk/EqB3QrnNG5FMpGzENQq_hBMBSaCQiZXP7yGCVlBHIGuSVA?e=UaSQCC). 

For a quick demo, please download ```blurball``` data inside ```real_camera_motion_blur``` folder.

### 3. Setting parameters
Changing the data path and log path in the ```configs/demo_blurball.txt```

### 4. Execute

```
python3 run_nerf.py --config configs/demo_blurball.txt
```
This will generate MLP weights and intermediate results in ```<basedir>/<expname>```, and generate tensorboard files in ```<tbdir>```

## Some Notes

### GPU memory
One drawback of our method is the large memory usage. Since we need to render multiple rays to generate only one color, we require a lot more memory to do that. We train our model on a ```V100``` GPU with 32GB GPU memory. If you have less memory, setting ```N_rand``` to a smaller value, or use multiple GPUs.

### Multiple GPUs
you can simply set ```num_gpu = <num_gpu>``` to use multiple gpus. It is implemented using ```torch.nn.DataParallel```. We've optimized the code to reduce the data transfering in each iteration, but it may still suffer from low GPU usable if you use too much GPU.

### Optimizing the CRF
As described in the paper, we use gamma function to model the nonlinearity of the blur kernel convolution (i.e. CRF). We provide an optional solution to optimize a learnable CRF function that transfer the color from linear space to the image color space. We've found that optimizing a learnable CRF sometimes achieves better visual performance. 

To make the CRF learnable, change the config to ```tone_mapping_type = learn```.

### Rollback to the original NeRF
By setting ```kernel_type = none```, our implementation runs the original implementation of NeRF.


### Your own data
The code digests the same data format as in the original implementation of NeRF (with the only difference being the images maybe blurry). So you can just use their data processing scripts to generate camera poses to ```poses_bounds.npy```.

Since all of our datasets are of type "llff" (face forward dataset), we do not support other dataset_type. But it should be relatively easy to migrate to other dataset_type. 

## Limitation
Our method does not work for consistent blur. For example, in the defocus blur case, if all the input views focus on the same foreground, leaving the background blur, our method only gives you NeRF with sharp foreground and blur background. Please check our paper for the definition of consistent blur. This is left for future work.

## Citation
If you find this useful, please consider citing our paper:
```
@misc{li2022deblurnerf,
    title={Deblur-NeRF: Neural Radiance Fields from Blurry Images},
    author={Ma, Li and Li, Xiaoyu and Liao, Jing and Zhang, Qi and Wang, Xuan and Wang, Jue and Pedro V. Sander},
    year={2021},
    eprint={2111.14292},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledge
This source code is derived from the famous pytorch reimplementation of NeRF, [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/). We appreciate the effort of the contributor to that repository.
