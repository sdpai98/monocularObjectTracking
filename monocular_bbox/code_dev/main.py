import sys
import os
from DepthModel import DepthModel
from DataManager import DataManager
from SimWorld import SimWorld
from Segmentation import Segmentation

import configuration

# dd3d imports
import logging
import os
from collections import OrderedDict, defaultdict
import pdb
import hydra
import torch
import wandb
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch.cuda import amp
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import detectron2.utils.comm as d2_comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluators, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, get_event_storage
from omegaconf import OmegaConf
import sys 
sys.path.append('/home/SamruddhiPai/Desktop/dd3d/')

import tridet.modeling  # pylint: disable=unused-import
import tridet.utils.comm as comm
from tridet.data import build_test_dataloader, build_train_dataloader
from tridet.data.dataset_mappers import get_dataset_mapper
from tridet.data.datasets import random_sample_dataset_dicts, register_datasets
from tridet.evaluators import get_evaluator
from tridet.modeling import build_tta_model
from tridet.utils.s3 import sync_output_dir_s3
from tridet.utils.setup import setup
from tridet.utils.train import get_inference_output_dir, print_test_results
from tridet.utils.visualization import mosaic, save_vis
from tridet.utils.wandb import flatten_dict, log_nested_dict
from tridet.visualizers import get_dataloader_visualizer, get_predictions_visualizer
import json
from omegaconf import OmegaConf


params_dict = {}
#Create data directory
data_manager = DataManager(configuration.ROOT_DIR)
capture_path = data_manager.get_capture_path()

# Configure depth model
depth_generator = DepthModel(configuration.DEPTH_MODEL)
depth_generator.configure_model()

# Create a world object
carla_world = SimWorld(configuration.WORLD_MAP, configuration.SCENE_WEATHER)

# Create a segmentation object
segmentation_model = Segmentation(configuration.SEGMENTATION_NETWORK)


from FrustumConvnet import FrustumConvnet
frustum_convnet = FrustumConvnet(folder=capture_path)

cfg = OmegaConf.load("/home/SamruddhiPai/Desktop/dd3d/configs/dd3d_config.yaml")
dd3d_model = build_model(cfg)
print('Built model')
checkpoint_file = cfg.MODEL.CKPT
if checkpoint_file:
    Checkpointer(dd3d_model).load(checkpoint_file)

print("Starting data capture")
carla_world.set_synchronous_mode(synch_mode = True, fps = configuration.FPS, no_render = False)
carla_world.spawn_ego_vehicle()
vehicles = carla_world.spawn_vehicles(configuration.NUM_VEHICLES)

print("Total number of vehicles spawned : ", len(vehicles))
carla_world.acquire_data(configuration.IMWIDTH, configuration.IMHEIGHT, configuration.IMFOV, configuration.FPS, configuration.NUM_FRAMES, depth_generator, segmentation_model, frustum_convnet, dd3d_model, capture_path, configuration.SHOW_IMAGE)