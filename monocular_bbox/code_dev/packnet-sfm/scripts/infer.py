# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch

from glob import glob
from cv2 import imwrite
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/packnet-sfm")

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor

#######################################################################
import kitti_util
import scipy.misc as ssc
#######################################################################


#######################################################################
def project_disp_to_points(calib, disp, max_high):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

def project_depth_to_points(calib, depth, max_high):
    print(depth.shape)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    pseudo_cloud_rect = calib.project_image_to_rect(points)
    cloud = calib.project_rect_to_velo(pseudo_cloud_rect)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return pseudo_cloud_rect, cloud[valid]
#######################################################################

def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png', 'npy'], default=None,
                        help='Save format (npz or png or npy). Default is None (no depth map is saved).')
    ##############################################################
    parser.add_argument('--calib_dir', type=str,
                        default='/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/KITTI_mini/object/training/calib')
    parser.add_argument('--disparity_dir', type=str,
                        default='/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/KITTI_mini/object/training/predicted_disparity')
    parser.add_argument('--save_dir', type=str,
                        default='/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/KITTI_mini/object/training/predicted_velodyne')
    parser.add_argument('--save_plot_dir', type=str,
                        default='/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/KITTI_mini/object/training/pseudo_lidar_plot')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--is_depth', action='store_true')
    parser.add_argument('--save_plot', type=bool, default=False)
    ###############################################################
    args = parser.parse_args()
    # args.image_shape = (1242,375)
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) and is_image(args.output)) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    ################################################################
    # assert os.path.isdir(args.disparity_dir)
    assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    #################################################################
    return args


@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, half, save):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    print('Inside infer depth',image_shape)
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    print('Shape of original image',image.size)
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)['inv_depths'][0]

    if save == 'npz' or save == 'png' or save == 'npy':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        depth_npy = inv2depth(pred_inv_depth)
        write_depth(filename, depth=inv2depth(pred_inv_depth))
        return depth_npy
    else:
        # Prepare RGB image
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        # Concatenate both vertically
        image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        imwrite(output_file, image[:, :, ::-1])
        # depth_npy = depth_npy.detach().squeeze().cpu().numpy()
        return image[:, :, ::-1]


def main(args):

    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    if os.path.isdir(args.input):
        # If input file is a folder, search for image files
        files = []
        for ext in ['png', 'jpg']:
            files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
        files.sort()
        print0('Found {} files'.format(len(files)))
    else:
        # Otherwise, use it as is
        files = [args.input]

    # Process each file
    for fn in files[rank()::world_size()]:
        depth = infer_and_save_depth(
            fn, args.output, model_wrapper, image_shape, args.half, args.save)
        
        # disps = [x for x in os.listdir(args.disparity_dir) if x[-3:] == 'png' or x[-3:] == 'npy']
        # disps = sorted(disps)
        ###############################################################################

        #predix = fn[:-4]
        predix = os.path.split(fn)[-1].split('.')[0]
        print(args.calib_dir, os.path.split(fn)[-1].split('.')[0])
        calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)
        # disp_map = ssc.imread(args.disparity_dir + '/' + fn) / 256.
        # if fn[-3:] == 'png':
        #     disp_map = ssc.imread(args.disparity_dir + '/' + fn)
        # elif fn[-3:] == 'npy':
        #     disp_map = np.load(args.disparity_dir + '/' + fn)
        # else:
        #     assert False
        # if not args.is_depth:
        #     disp_map = (disp_map*256).astype(np.uint16)/256.
        #     lidar = project_disp_to_points(calib, disp_map, args.max_high)
        # else:
        if torch.is_tensor(depth):
          depth = depth.detach().squeeze().cpu().numpy()
        disp_map = (depth).astype(np.float32)/256.
        pseudo_lidar, lidar = project_depth_to_points(calib, disp_map, args.max_high)
        # pad 1 in the indensity dimension
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        print('Shape of final depth', depth.shape)
        depth.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish Depth {}'.format(predix))

        #3D scatter plot of pseudo lidar point cloud
        if args.save_plot:
          fig = plt.figure()
          ax = fig.add_subplot(projection='3d')

          xs = pseudo_lidar[:, 0] * 255
          ys = pseudo_lidar[:, 1] * 255
          zs = pseudo_lidar[:, 2] * 255

          ax.scatter(xs, ys, zs)
          pseudo_lidar = np.reshape(pseudo_lidar, (disp_map.shape[0], disp_map.shape[1], 3))
          fig.savefig('{}/{}.png'.format(args.save_plot_dir, predix))

        # np.save('{}/{}.npy'.format(args.save_dir, predix), pseudo_lidar)
        # print('Finish Depth {}'.format(predix))
    ###############################################################################


if __name__ == '__main__':
    args = parse_args()
    main(args)
