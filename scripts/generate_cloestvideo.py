import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os
import imageio
import sys
import argparse

from numpy import random
sys.path.append('D:\\MSI_NB\\source\\repos\\nerf-pytorch')
# from NeRF import *
from load_llff import load_llff_data
# from run_nerf_helpers import *


datadir = "defocustools"
outdir = "Z:\\NeRF_material\\material_supplementary\\nearest_video"
factor = 4
llffhold = 7

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.render_focuspoint_scale = 1.
args.render_radius_scale = 1.
images, poses, bds, render_poses, i_test = load_llff_data(args, datadir, factor, 
                                                          recenter=True, bd_factor=.75,
                                                          spherify=False)
cam_center = render_poses[:, :3, :3] @ render_poses[:, :3, 3:4]
source_center = poses[:, :3, :3] @ poses[:, :3, 3:4]

distances = cam_center[:, None, :, 0] - source_center[None, :, :, 0]
distances = np.linalg.norm(distances, axis=-1)
distances[:, ::llffhold] = 1e10
index = np.argmin(distances, axis=-1)

images = (images * 255).astype(np.uint8)
video = [images[i] for i in index]

os.makedirs(outdir, exist_ok=True)
imageio.mimwrite(f"{outdir}/{os.path.basename(datadir)}_nearest.mp4", video, fps=30, quality=8)
