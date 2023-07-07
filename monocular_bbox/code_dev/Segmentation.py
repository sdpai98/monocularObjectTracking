import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import some common libraries
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import numpy as np
import time
import math
import torch.nn as nn
import copy

import configuration as config

class Segmentation():
	def __init__(self, network):
		self.network = network
		self.predictor = None
		self.configure_model()
		

	def configure_model(self):
		if(self.network == 'detectron'):
			cfg = get_cfg()
			cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
			cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
			cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

			self.predictor = DefaultPredictor(cfg)

		elif(self.network == 'maskrcnn'):
			model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
			model.eval()
			self.predictor = model

	def get_bounding_boxes(self, img):
		if(self.network == 'detectron'):
			outputs = self.predictor(img)
			boxes = outputs['instances'].pred_boxes.tensor.cpu().detach().numpy()
			num_instances = len(boxes)
			classes = outputs['instances'].pred_classes.cpu().detach()
			classes = [config.pred_classes_mapping[i+1] for i in classes]
			scores = outputs['instances'].scores.cpu().detach()

		elif(self.network == 'maskrcnn'):
			trans_img = np.transpose(img, (2, 0, 1))
			tens = torch.from_numpy(trans_img)
			img_tensor = tens.type(torch.FloatTensor)
			img_tensor = img_tensor/255.0
			preds = self.predictor([img_tensor])

			boxes = preds[0]['boxes']
			classes = preds[0]['labels']
			num_instances = len(classes)
			classes = [config.pred_classes_mapping[i] for i in classes]
			scores = preds[0]['scores']

		boxes_car = []
		classes_car = []
		scores_car = []

		for i in range(num_instances):
			if classes[i] == 'Car' and scores[i] > config.SEGMENTATION_THRESH:
				boxes_car.append(boxes[i])
				classes_car.append(classes[i])
				scores_car.append(scores[i])

		return len(boxes_car), classes_car, boxes_car, scores_car




