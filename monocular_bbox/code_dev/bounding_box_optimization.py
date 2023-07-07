# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions for calculating 2D and 3D bounding box IoU.
Collected and written by Charles R. Qi
Last modified: Jul 2019
"""

import torch
import torchvision
import cv2
import numpy as np
import time
import math
import sys
import torch.nn as nn
sys.path.append('/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/frustum-convnet/kitti')

import kitti_util as util


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l,w,h = box_size
    # x_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    # y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    # z_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    # print('Shape of corners_3d',corners_3d.shape)
    corners_3d = np.transpose(np.vstack([corners_3d,np.ones((1,8))]))
    # print('Shape of corners_3d',corners_3d.shape)
    return corners_3d


def get_coordinates(n):
  # print(((-1)**(n)))
  x = x_c + ((-1)**(n))*del_x + ((-1)**int(n/4))*del_theta
  y = y_c + ((-1)**int(n/2))*del_y
  z = z_c + ((-1)**int(n/4))*del_z
  # print(x,y,z)
  return [x,y,z,1]


def mbr(arr):
  x_min = min(arr[:,0])
  x_max = max(arr[:,0])
  y_min = min(arr[:,1])
  y_max = max(arr[:,1])
  return torch.FloatTensor([x_min,y_min,x_max,y_max])


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # print('Estimated', boxA)
    # print('Predicted', boxB)
    # print('intersection points',xA,xB,yA,yB)
    # compute the area of intersection rectangle
    # interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    interArea = abs(xB-xA)*abs(yB-yA)
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bbox_optimization(bbox_2d, box_size, center, heading_angle, calib):
    
    # print('Original theta value with noise - ', heading_angle)

    min_loss = 10000
    max_iou = 0
    t = heading_angle-0.5 # starting with an offset of -20 deg
    mbr_p = bbox_2d
    best_ry = t
    best_mbr = np.array([10,10,100,100])
    # print('Calib in bbox_optimization', calib)
    for j in range(10):
        # import pdb; pdb.set_trace()
        coord_3d = get_3d_box(box_size, t, center)
        coord_2D = np.dot(coord_3d,calib['P2'].T)    
        #   coord_2D = np.transpose(coord_2D)
        # print('coord_2d shape',coord_2D.shape)
        coord_2D[:,0] = coord_2D[:,0]/coord_2D[:,2]
        coord_2D[:,1] = coord_2D[:,1]/coord_2D[:,2]
        mbr_e = mbr(coord_2D)
        #   mbr_e_mean = torch.mean(mbr_e)
        #   if j == 1:
        #     print('mbr_e',mbr_e)
        #     l1 = bb_intersection_over_union(mbr_e, mbr_p)
        #     initial_loss.append(l1)
        #     cv2.rectangle(optimized_img_output,(mbr_e[0],mbr_e[1]-25),(mbr_e[2],mbr_e[3]-35),(0,255,0),2)
        #     cv2.imwrite('optimized_img_output.jpg',optimized_img_output)   
        # l1 = loss(mbr_e, mbr_p)
        l1 = bb_intersection_over_union(mbr_e, mbr_p)
        # print(l1)
        # if l1<min_loss:
        if l1>max_iou:
            best_ry = t
            # min_loss = l1
            max_iou = l1
            # print('current max IOU!!!', max_iou)
            best_mbr = mbr_e
        t = t+0.1
    # print('final value of theta', best_ry)
    return best_ry, best_mbr

def bbco(params_3d, bbox_2d, center, calib):
    loss = nn.SmoothL1Loss(beta=1.0)
    h,w,l,theta = params_3d
    coord_3d = get_3d_box([l,w,h], theta, center)
    coord_2D = np.dot(coord_3d,calib['P2'].T)    
    coord_2D[:,0] = coord_2D[:,0]/coord_2D[:,2]
    coord_2D[:,1] = coord_2D[:,1]/coord_2D[:,2]
    mbr_e = mbr(coord_2D)
    iou = bb_intersection_over_union(mbr_e, bbox_2d)
    # print(np.array(mbr_e),np.array(bbox_2d))
    # print(type(mbr_e),type(bbox_2d))
    # l1_loss = loss(mbr_e, torch.from_numpy(bbox_2d))
    # print('Loss!!!!!!',l1_loss)
    # return l1_loss
    return 1/iou