''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017

Modified by Zhixin Wang
'''

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys
sys.path.append("/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/frustum-convnet")
sys.path.append("/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/frustum-convnet/kitti")

import argparse
import os
import pickle

import cv2
import numpy as np
from PIL import Image
import kitti_util

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
# sys.path.append(ROOT_DIR)


# import kitti_util as utils
# from kitti_object import kitti_object
# from draw_util import get_lidar_in_image_fov

from ops.pybind11.rbbox_iou import bbox_overlaps_2d

import configuration

import os
import sys
import math
import shutil
import time
import argparse

import pprint
import random as pyrandom
import logging
# from mayavi import mlab
import pptk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import pickle
import subprocess
# from pcvisualizer import pc_viz

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(ROOT_DIR)

sys.path.append("/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/frustum-convnet")
from configs_1.config import cfg
from configs_1.config import merge_cfg_from_file
from configs_1.config import merge_cfg_from_list
from configs_1.config import assert_and_infer_cfg

from utils.training_states import TrainingStates
from utils.utils import get_accuracy, AverageMeter, import_from_file, get_logger

from datasets.provider_sample import from_prediction_to_label_format, compute_alpha
from datasets.dataset_info import DATASET_INFO

from ops.pybind11.rbbox_iou import cube_nms_np
from ops.pybind11.rbbox_iou import bev_nms_np
from ops.pybind11.rbbox_iou import rotate_nms_3d_cc as cube_nms
from ops.pybind11.rbbox_iou import rotate_nms_bev_cc as bev_nms

sys.path.append("/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/frustum-convnet/datasets")
from provider_sample import ProviderDataset

from bounding_box_optimization import bbox_optimization, bbco, get_3d_box, mbr
from scipy.optimize import differential_evolution

# def draw_rectangle(img, boxes):
#     rmask_rcnn_image = copy.deepcopy(img)
#     for i in boxes:
#         cv2.rectangle(rmask_rcnn_image,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(0,255,0),1)
# 	cv2.imshow('projected 2d box',img)

class FrustumConvnet:

	def __init__(self):
		pass

	def get_lidar_in_image_fov(self, pc_velo, calib, xmin, ymin, xmax, ymax,
						   return_more=False, clip_distance=2.0):

		pts_2d = pc_velo[:, :3] #calib.project_velo_to_image(pc_velo[:, :3])
		fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
			(pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
		fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
		imgfov_pc_velo = pc_velo[fov_inds, :]
		if return_more:
			return imgfov_pc_velo, pts_2d, fov_inds
		else:
			return imgfov_pc_velo

	def project_image_to_rect(self, uv_depth, calib_P2):
		''' Input: nx3 first two channels are uv, 3rd channel
				   is depth in rect camera coord.
			Output: nx3 points in rect camera coord.
		'''
		# calib_P2 = calib_P2['P2']
		# Camera intrinsics and extrinsics
		self.c_u = calib_P2[0, 2]
		self.c_v = calib_P2[1, 2]
		self.f_u = calib_P2[0, 0]
		self.f_v = calib_P2[1, 1]
		self.b_x = calib_P2[0, 3] / (-self.f_u)  # relative
		self.b_y = calib_P2[1, 3] / (-self.f_v)

		n = uv_depth.shape[0]
		x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
		y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
		pts_3d_rect = np.zeros((n, 3))
		pts_3d_rect[:, 0] = x
		pts_3d_rect[:, 1] = y
		pts_3d_rect[:, 2] = uv_depth[:, 2]

		return pts_3d_rect


	def set_random_seed(self, seed=3):
		pyrandom.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

	def homogeneous_transform(self, points, transform):
		"""
		Parameters
		----------
		points : (n_points, M) array-like
			The points to transform. If `points` is shape (n_points, M-1), a unit
			homogeneous coordinate will be added to make it (n_points, M).
		transform : (M, N) array-like
			The right-multiplying transformation to apply.
		"""
		points = np.asarray(points)
		transform = np.asarray(transform)
		n_points, D = points.shape
		M, N = transform.shape

		print('Shapes transform, points', transform.shape,  points.shape)
		# do transformation in homogeneous coordinates
		if D == M - 1:
			points = np.hstack([points, np.ones((n_points, 1), dtype=points.dtype)])
		elif D != M:
			raise ValueError("Number of dimensions of points (%d) does not match"
								"input dimensions of transform (%d)." % (D, M))

		new_points = np.dot(points, transform)
		print('new_ppoints shape', new_points.shape)
		# normalize homogeneous coordinates
		new_points = new_points / new_points[:, -1].reshape(-1,1)
		return new_points

	def get_rect2disp(self, calib):
		# cam0, cam1 = (0, 1) if not self.color else (2, 3)
		P_rect0 = calib['P2'].reshape(3, 4)
		P_rect1 = calib['P3'].reshape(3, 4)

		P0, P1, P2 = P_rect0
		Q0, Q1, Q2 = P_rect1
		# assert np.array_equal(P1, Q1), "\n%s\n%s" % (P1, Q1)
		# assert np.array_equal(P2, Q2), "\n%s\n%s" % (P2, Q2)

		# create disp transform
		T = np.array([P0, P1, P0 - Q0, P2])
		return T.T

	def pc_viz(self,pc):
		z_th = 0.2
		x = pc[:,0]


		y = pc[:,1]


		z = pc[:,2]

		indices = np.where(z < z_th)
		print((np.max(z), np.min(z)))
		xs = x[indices]
		ys = y[indices]
		zs = z[indices]
		return pc[indices,:3]
	
	def print_min_max(self,points):
		print('Xc', np.min(points[:,0]),np.max(points[:,0]))
		print('Yc', np.min(points[:,1]),np.max(points[:,1]))
		print('Zc', np.min(points[:,2]),np.max(points[:,2]))
		if points.shape[1]>3:
			print('homo', np.min(points[:,3]),np.max(points[:,3]))


	def prepare_data(self, det_id_list, det_type_list, det_box2d_list, det_prob_list, \
		output_filename, calib_P2, pc_velo, img_height_threshold=5, lidar_point_threshold=1):

		self.id_list = []
		self.type_list = []
		self.box2d_list = []
		self.prob_list = []
		self.input_list = []  # channel number = 4, xyz,intensity in rect camera coord
		self.frustum_angle_list = []  # angle of 2d box center from pos x-axis
		self.calib_list = []
		#### Verification #####

		# RT_velo2cam = np.eye(4)
	
		# RT_velo2cam = calib_P2['Tr_velo_to_cam'].reshape(4, 4)
		
		# # R_rect00 = np.eye(4)
		# R_rect00 = calib_P2['R0_rect'].reshape(4, 4)

		# RT_velo2rect = np.dot(R_rect00, RT_velo2cam).T

		'''
		points = np.fromfile('/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/KITTI_mini/object/testing/velodyne/000002.bin', dtype=np.float32).reshape(-1, 4)
		lidar_org = points[:, :3]
		v1 = pptk.viewer(lidar_org)
		v1.wait()
		print('org lidar shape!!!!',lidar_org.shape)
		print('Xc', np.min(lidar_org[:,0]),np.max(lidar_org[:,0]))
		print('Yc', np.min(lidar_org[:,1]),np.max(lidar_org[:,1]))
		print('Zc', np.min(lidar_org[:,2]),np.max(lidar_org[:,2]))

		pt_velo2rect = self.homogeneous_transform(lidar_org, RT_velo2rect)

		v1 = pptk.viewer(pt_velo2rect[:, :3])
		v1.wait()

		print('velo to rect shape!!!!',pt_velo2rect.shape)
		print('Xc', np.min(pt_velo2rect[:,0]),np.max(pt_velo2rect[:,0]))
		print('Yc', np.min(pt_velo2rect[:,1]),np.max(pt_velo2rect[:,1]))
		print('Zc', np.min(pt_velo2rect[:,2]),np.max(pt_velo2rect[:,2]))
		# print('homo', np.min(pt_velo2rect[:,3]),np.max(pt_velo2rect[:,3]))

		# pt_rect2disp = self.homogeneous_transform(points, self.get_rect2disp(calib_P2))

		# v1 = pptk.viewer(pt_rect2disp)
		# v1.wait()

		#### Verification #####
		'''
		# print('shape of pc_velo',pc_velo.shape)
		pc_velo[:, 2] = pc_velo[:, 2]*15 # multiplying depth by 256
		pc_rect = np.zeros_like(pc_velo)
		# v2 = pptk.viewer(pc_velo[:,:3],-pc_velo[:,2])
		# v2.wait()
		# print('Xc', np.min(pc_velo[:,0]), np.max(pc_velo[:,0]))
		# print('Yc', np.min(pc_velo[:,1]), np.max(pc_velo[:,1]))
		# print('Zc', np.min(pc_velo[:,2]), np.max(pc_velo[:,2]))
		# print('homo', np.min(pc_velo[:,3]), np.max(pc_velo[:,3]))

		# idx = np.random.choice(pc_velo.shape[0],100000)
		# pc_filtered = pc_velo[idx,:]
		# print('pc_filtered',pc_filtered.shape)
		# v = pptk.viewer(pc_filtered[:,:3])
		# v.wait()

		pc_rect[:, 0:3] = self.project_image_to_rect(pc_velo[:, 0:3], calib_P2['P2']) # real world --> pc_rect, performs math given in Kitana to get (Xc, Yc, Zc)
		pc_rect[:, 3] = pc_velo[:, 3]
		# v3 = pptk.viewer(pc_rect[:,:3],-pc_rect[:,2])
		# v3.wait()
		# print('Image to rect shape!!!!',pc_rect.shape)
		# print('Xc', np.min(pc_rect[:,0]),np.max(pc_rect[:,0]))
		# print('Yc', np.min(pc_rect[:,1]),np.max(pc_rect[:,1]))
		# print('Zc', np.min(pc_rect[:,2]),np.max(pc_rect[:,2]))
		# print('homo', np.min(pc_rect[:,3]),np.max(pc_rect[:,3]))

		# pt_rect2velo = self.homogeneous_transform(pc_rect, np.linalg.inv(RT_velo2rect))
		# print('rect to velo shape!!!!',pt_rect2velo.shape)
		# self.print_min_max(pt_rect2velo)

		# v1 = pptk.viewer(pt_rect2velo[:, :3],-pt_rect2velo[:,0])
		# v1.wait()

		img_width = configuration.IMWIDTH
		img_height = configuration.IMHEIGHT

		img_fov_lidar, pc_image_coord, img_fov_inds = self.get_lidar_in_image_fov(
			pc_velo[:, 0:3], calib_P2, 0, 0, img_width, img_height, True)
		# cache = [calib, pc_rect, pc_image_coord, img_fov_inds]
		# cache_id = data_idx
		# pc_image_coord = pc_velo[:, 0:3]
		# v4 = pptk.viewer(img_fov_lidar[:,:3])
		# v4.wait()
		# print('img fov lidar shape!!!!',img_fov_lidar.shape)
		# print('Xc', np.min(img_fov_lidar[:,0]),np.max(img_fov_lidar[:,0]))
		# print('Yc', np.min(img_fov_lidar[:,1]),np.max(img_fov_lidar[:,1]))
		# print('Zc', np.min(img_fov_lidar[:,2]),np.max(img_fov_lidar[:,2]))
		# # print('homo', np.min(img_fov_lidar[:,3]),np.max(img_fov_lidar[:,3]))

		# sys.exit()

		for det_idx in range(det_id_list):
			# data_idx = det_id_list[det_idx]

			# if det_type_list[det_idx] not in type_whitelist:
			#     continue

			# 2D BOX: Get pts rect backprojected
			det_box2d = det_box2d_list[det_idx].copy()
			det_box2d[[0, 2]] = np.clip(det_box2d[[0, 2]], 0, img_width - 1)
			det_box2d[[1, 3]] = np.clip(det_box2d[[1, 3]], 0, img_height - 1)

			xmin, ymin, xmax, ymax = det_box2d
			box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
						   (pc_image_coord[:, 0] >= xmin) & \
						   (pc_image_coord[:, 1] < ymax) & \
						   (pc_image_coord[:, 1] >= ymin)
			box_fov_inds = box_fov_inds & img_fov_inds
			pc_in_box_fov = pc_rect[box_fov_inds, :]

			pc_box_image_coord = pc_image_coord[box_fov_inds, :]

			# Get frustum angle (according to center pixel in 2D BOX)
			box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
			uvdepth = np.zeros((1, 3))
			uvdepth[0, 0:2] = box2d_center
			uvdepth[0, 2] = 20  # some random depth
			# print('calib_p2',calib_P2 )
			box2d_center_rect = self.project_image_to_rect(uvdepth, calib_P2['P2'])
			frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
											box2d_center_rect[0, 0])

			# Pass objects that are too small
			if ymax - ymin < img_height_threshold or xmax - xmin < 1 or \
					len(pc_in_box_fov) < lidar_point_threshold:
				continue

			self.id_list.append(det_idx)
			self.type_list.append(det_type_list[det_idx])
			self.box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
			self.prob_list.append(det_prob_list[det_idx])
			self.input_list.append(pc_in_box_fov.astype(np.float32, copy=False))
			self.frustum_angle_list.append(frustum_angle)
			self.calib_list.append(calib_P2)

		# with open(output_filename, 'wb') as fp:
		#     pickle.dump(id_list, fp, -1)
		#     pickle.dump(box2d_list, fp, -1)
		#     pickle.dump(input_list, fp, -1)
		#     pickle.dump(type_list, fp, -1)
		#     pickle.dump(frustum_angle_list, fp, -1)
		#     pickle.dump(prob_list, fp, -1)
		#     pickle.dump(calib_list, fp, -1)

		# print('total_objects %d' % len(id_list))
		# print('save in {}'.format(output_filename))

	def fill_files(self, output_dir, to_fill_filename_list):
		''' Create empty files if not exist for the filelist. '''
		for filename in to_fill_filename_list:
			filepath = os.path.join(output_dir, filename)
			if not os.path.exists(filepath):
				fout = open(filepath, 'w')
				fout.close()

	def write_detection_results(self, output_dir, det_results, frame_id=0, kitti=True):

		results = {}  # map from idx to list of strings, each string is a line (without \n)
		for idx in det_results:
			for class_type in det_results[idx]:
				dets = det_results[idx][class_type]
				for i in range(len(dets)):
					box2d = dets[i][:4]
					tx, ty, tz, h, w, l, ry = dets[i][4:-1]
					score = dets[i][-1]
					alpha = compute_alpha(tx, tz, ry)
					output_str = class_type + " -1 -1 "
					output_str += '%.4f ' % alpha
					output_str += "%.4f %.4f %.4f %.4f " % (box2d[0], box2d[1], box2d[2], box2d[3])
					output_str += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %f" % (h, w, l, tx, ty, tz, ry, score)
					if idx not in results:
						results[idx] = []
					results[idx].append(output_str)

		result_dir = os.path.join(output_dir, 'data')
		# os.system('rm -rf %s' % (result_dir))
		if not os.path.exists(result_dir):
			os.mkdir(result_dir)

		# print("##### RESULTS #######")
		# print(results)
		# print("#####################")

		# return results
		if kitti:
			pred_filename = os.path.join(result_dir, '%06d.txt' % (frame_id))
			# print('kitti\'s output file name:',pred_filename)
			fout = open(pred_filename, 'w')
			# fout.write('Writing results!!!!!!')
			bboxes_3d = []
			for idx in results:
				line = results[idx][0]
				print('line!!!!',line)
				bboxes_3d.append(kitti_util.Object3d(line))
				fout.write(line + '\n')
			fout.close()	
		else:
			for idx in results:
				pred_filename = os.path.join(result_dir, '%06d.txt' % (idx))
				print('output file name:',pred_filename)
				fout = open(pred_filename, 'w')
				for line in results[idx]:
					fout.write(line + '\n')
				fout.close()

		# Make sure for each frame (no matter if we have measurement for that frame),
		# there is a TXT file
		idx_path = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/frustum-convnet/kitti/image_sets/%s.txt' % cfg.TEST.DATASET

		to_fill_filename_list = [line.rstrip() + '.txt' for line in open(idx_path)]
		self.fill_files(result_dir, to_fill_filename_list)
		return results, bboxes_3d

	def write_detection_results_nms(self, output_dir, det_results, threshold, frame_id=0, kitti=False):

		nms_results = {}
		for idx in det_results:
			for class_type in det_results[idx]:
				dets = np.array(det_results[idx][class_type], dtype=np.float32)
				# scores = dets[:, -1]
				# keep = (scores > 0.001).nonzero()[0]
				# print(len(scores), len(keep))
				# dets = dets[keep]
				if len(dets) > 1:
					# (tx, ty, tz, h, w, l, ry, score) -> (tx, ty, tz, l, w, h, ry, score)
					dets_for_nms = dets[:, 4:][:, [0, 1, 2, 5, 4, 3, 6, 7]]
					# keep = cube_nms_np(dets_for_nms, threshold)
					keep = cube_nms(dets_for_nms, threshold)
					# (tx, ty, tz, h, w, l, ry, score) -> (tx, tz, l, w, ry, score)
					# dets_for_bev_nms = dets[:, 4:][:, [0, 2, 5, 4, 6, 7]]
					# keep = bev_nms_np(dets_for_bev_nms, threshold)
					# keep = bev_nms(dets_for_bev_nms, threshold)
					dets_keep = dets[keep]
				else:
					dets_keep = dets
				if idx not in nms_results:
					nms_results[idx] = {}
				nms_results[idx][class_type] = dets_keep

		return self.write_detection_results(output_dir, nms_results, frame_id, kitti)


	def test(self):

		
		#args = parse_args()

		result_dir = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/output'
		cfg_file = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/frustum-convnet/cfgs/det_sample.yaml'
		opts = ['OUTPUT_DIR', 'pretrained_models/car', 'TEST.WEIGHTS', \
		'/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/pretrained_models/car/model_0050.pth']
		
		if cfg_file is not None:
			merge_cfg_from_file(cfg_file)

		if opts is not None:
			merge_cfg_from_list(opts)

		assert_and_infer_cfg()

		SAVE_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.SAVE_SUB_DIR)

		if not os.path.exists(SAVE_DIR):
			os.makedirs(SAVE_DIR)

		# set logger
		cfg_name = os.path.basename(cfg_file).split('.')[0]
		log_file = '{}_{}_val.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))

		logger = get_logger(os.path.join(SAVE_DIR, log_file))
		# logger.info('config:\n {}'.format(pprint.pformat(cfg)))

		model_def = import_from_file(cfg.MODEL.FILE)
		model_def = model_def.PointNetDet

		dataset_def = import_from_file(cfg.DATA.FILE)
		collate_fn = dataset_def.collate_fn
		dataset_def = dataset_def.ProviderDataset

		# overwritten_data_path = None
		# if cfg.OVER_WRITE_TEST_FILE and cfg.FROM_RGB_DET:
		#     overwritten_data_path = cfg.OVER_WRITE_TEST_FILE
		print('Before test dataset',cfg.OVER_WRITE_TEST_FILE)
		test_dataset = dataset_def(
			cfg.DATA.NUM_SAMPLES,
			split=cfg.TEST.DATASET,
			random_flip=False,
			random_shift=False,
			one_hot=True,
			from_rgb_detection=cfg.FROM_RGB_DET,
			overwritten_data_path=cfg.OVER_WRITE_TEST_FILE,
			extend_from_det=False,
			id_list = self.id_list, 
			type_list = self.type_list, 
			box2d_list = self.box2d_list, 
			prob_list = self.prob_list, 
			input_list = self.input_list, 
			frustum_angle_list = self.frustum_angle_list, 
			calib_list = self.calib_list)

		test_loader = torch.utils.data.DataLoader(
			test_dataset,
			batch_size=cfg.TEST.BATCH_SIZE,
			shuffle=False,
			num_workers=cfg.NUM_WORKERS,
			pin_memory=True,
			drop_last=False,
			collate_fn=collate_fn)

		input_channels = 3 if not cfg.DATA.WITH_EXTRA_FEAT else cfg.DATA.EXTRA_FEAT_DIM
		# NUM_VEC = 0 if cfg.DATA.CAR_ONLY else 3
		# NUM_VEC = 3
		dataset_name = cfg.DATA.DATASET_NAME
		assert dataset_name in DATASET_INFO
		datset_category_info = DATASET_INFO[dataset_name]
		NUM_VEC = len(datset_category_info.CLASSES) # rgb category as extra feature vector
		NUM_CLASSES = cfg.MODEL.NUM_CLASSES

		model = model_def(input_channels, num_vec=NUM_VEC, num_classes=NUM_CLASSES)

		model = model.cuda()

		print("!!!!!!!!!!", cfg.TEST.WEIGHTS)
		if os.path.isfile(cfg.TEST.WEIGHTS):
			checkpoint = torch.load(cfg.TEST.WEIGHTS)
			# start_epoch = checkpoint['epoch']
			# best_prec1 = checkpoint['best_prec1']
			# best_epoch = checkpoint['best_epoch']
			if 'state_dict' in checkpoint:
				model.load_state_dict(checkpoint['state_dict'])
				logging.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.TEST.WEIGHTS, checkpoint['epoch']))
			else:
				model.load_state_dict(checkpoint)
				logging.info("=> loaded checkpoint '{}')".format(cfg.TEST.WEIGHTS))
		else:
			logging.error("=> no checkpoint found at '{}'".format(cfg.TEST.WEIGHTS))
			assert False

		if cfg.NUM_GPUS > 1:
			model = torch.nn.DataParallel(model)

		save_file_name = os.path.join(SAVE_DIR, 'detection.pkl')
		result_folder = os.path.join(SAVE_DIR, 'result')

		if not os.path.exists(result_folder):
			os.makedirs(result_folder)

		#----------------------------- BEGINNING OF TEST FUNCTION -----------------------
		load_batch_size = test_loader.batch_size
		num_batches = len(test_loader)

		model.eval()

		fw_time_meter = AverageMeter()

		det_results = {}

		for i, data_dicts in enumerate(test_loader):
			# print('Data_dict',data_dicts)
			point_clouds = data_dicts['point_cloud']
			rot_angles = data_dicts['rot_angle']
			# optional
			ref_centers = data_dicts.get('ref_center')
			rgb_probs = data_dicts.get('rgb_prob')

			# from ground truth box detection
			if rgb_probs is None:
				rgb_probs = torch.ones_like(rot_angles)

			# not belong to refinement stage
			if ref_centers is None:
				ref_centers = torch.zeros((point_clouds.shape[0], 3))

			batch_size = point_clouds.shape[0]
			rot_angles = rot_angles.view(-1)
			rgb_probs = rgb_probs.view(-1)

			if 'box3d_center' in data_dicts:
				data_dicts.pop('box3d_center')

			data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}

			torch.cuda.synchronize()
			tic = time.time()
			with torch.no_grad():
				outputs = model(data_dicts_var)

			# cls_probs, center_preds, heading_preds, size_preds = outputs
			cls_probs, center_preds, heading_preds, size_preds, heading_probs, size_probs = outputs

			torch.cuda.synchronize()
			fw_time_meter.update((time.time() - tic))

			num_pred = cls_probs.shape[1]
			print('%d/%d %.3f' % (i, num_batches, fw_time_meter.val))

			cls_probs = cls_probs.data.cpu().numpy()
			center_preds = center_preds.data.cpu().numpy()
			heading_preds = heading_preds.data.cpu().numpy()
			size_preds = size_preds.data.cpu().numpy()
			heading_probs = heading_probs.data.cpu().numpy()
			size_probs = size_probs.data.cpu().numpy()

			rgb_probs = rgb_probs.numpy()
			rot_angles = rot_angles.numpy()
			ref_centers = ref_centers.numpy()

			for b in range(batch_size):

				if cfg.TEST.METHOD == 'nms':
					fg_idx = (cls_probs[b, :, 0] < cls_probs[b, :, 1]).nonzero()[0]
					if fg_idx.size == 0:
						fg_idx = np.argmax(cls_probs[b, :, 1])
						fg_idx = np.array([fg_idx])
				else:
					fg_idx = np.argmax(cls_probs[b, :, 1])
					fg_idx = np.array([fg_idx])

				num_pred = len(fg_idx)

				single_centers = center_preds[b, fg_idx]
				single_headings = heading_preds[b, fg_idx]
				single_sizes = size_preds[b, fg_idx]
				single_scores = cls_probs[b, fg_idx, 1] + rgb_probs[b]

				data_idx = test_dataset.id_list[load_batch_size * i + b]
				class_type = test_dataset.type_list[load_batch_size * i + b]
				box2d = test_dataset.box2d_list[load_batch_size * i + b]
				rot_angle = rot_angles[b]
				ref_center = ref_centers[b]
				calib = test_dataset.calib_list[load_batch_size * i + b]

				if data_idx not in det_results:
					det_results[data_idx] = {}

				if class_type not in det_results[data_idx]:
					det_results[data_idx][class_type] = []

				for n in range(num_pred):
					x1, y1, x2, y2 = box2d
					score = single_scores[n]
					h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(
						single_centers[n], single_headings[n], single_sizes[n], rot_angle, ref_center)
					# filter out too small object, although it is impossible  in most casts
					if h < 0.01 or w < 0.01 or l < 0.01:
						continue

					coord_3d = get_3d_box([l,w,h], ry, [tx,ty,tz])
					coord_2D = np.dot(coord_3d,calib['P2'].T)    
					coord_2D[:,0] = coord_2D[:,0]/coord_2D[:,2]
					coord_2D[:,1] = coord_2D[:,1]/coord_2D[:,2]
					pred_2dbox = mbr(coord_2D)

					actual_area = abs(x1-x2)*abs(y1-y2)
					pred_area = abs(pred_2dbox[0]-pred_2dbox[2])*abs(pred_2dbox[1]-pred_2dbox[3])
					print('pred 2d box', pred_2dbox)
					print('Actual 2d box',box2d)
					print('difference in the area', abs(actual_area-pred_area))
					bounds = [[max(h-0.5,0.0),h+0.5],[max(w-0.5,0.0),w+0.5],[max(l-1,0.0),l+1],[max(ry-0.5,-3.14),min(ry+0.5,3.14)]] # ranges for h,w,l,theta
					result = differential_evolution(bbco, bounds, args=(box2d, [tx,ty,tz], calib))
					print('Bounds', bounds)
					print('After optimization, iou = ',1/result.fun) 
					# sys.exit()
					h,w,l,ry = list(result.x)
					ry, best_mbr = bbox_optimization([x1,y1,x2,y2],(l,w,h),(tx,ty,tz),ry, calib)
					output = [x1, y1, x2, y2, tx, ty, tz, h, w, l, ry, score]
					det_results[data_idx][class_type].append(output)

		num_images = len(det_results)

		# logging.info('Average time:')
		# logging.info('batch:%0.3f' % fw_time_meter.avg)
		# logging.info('avg_per_object:%0.3f' % (fw_time_meter.avg / load_batch_size))
		# logging.info('avg_per_image:%.3f' % (fw_time_meter.avg * len(test_loader) / num_images))

		# Write detection results for KITTI evaluation

		# print("det_results", det_results)
		

		# if cfg.TEST.METHOD == 'nms':
		# 	print('In nms loop!!!')
		# 	det_results_final = self.write_detection_results_nms(result_dir, det_results, threshold=cfg.TEST.THRESH)
		# else:
		# 	det_results_final = self.write_detection_results(result_dir, det_results)

		# print("####### FINAL RESULTS #########")
		# print(det_results_final)
		# return det_results_final
		return det_results

		# output_dir = os.path.join(result_dir, 'data')

		# if 'test' not in cfg.TEST.DATASET:
		#     if os.path.exists('../kitti-object-eval-python'):
		#         evaluate_cuda_wrapper(output_dir, cfg.TEST.DATASET)
		#     else:
		#         evaluate_py_wrapper(result_dir)
			   
		# else:
		#     logger.info('results file save in  {}'.format(result_dir))
		#     os.system('cd %s && zip -q -r ../results.zip *' % (result_dir))

