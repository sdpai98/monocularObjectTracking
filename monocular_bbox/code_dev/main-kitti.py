import sys
import os
import glob
from pathlib import Path    
from data_processing import process_image_kitti, process_image
from DepthModel import DepthModel
# from DataManager import DataManager
# from SimWorld import SimWorld
from Segmentation import Segmentation
import configuration as configuration
from kitti_utils import LoadCalibrationFile 

params_dict = {}
#Create data directory
# data_manager = DataManager(configuration.ROOT_DIR)
# capture_path = data_manager.get_capture_path()

# Configure depth model
depth_generator = DepthModel(configuration.DEPTH_MODEL)
depth_generator.configure_model()

# Create a world object
# carla_world = SimWorld(configuration.WORLD_MAP, configuration.SCENE_WEATHER)

# Create a segmentation object
segmentation_model = Segmentation(configuration.SEGMENTATION_NETWORK)


from FrustumConvnet import FrustumConvnet
frustum_convnet = FrustumConvnet('/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/Data/kitti')

calibration = LoadCalibrationFile('/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/KITTI_mini/object/testing/calib/000000.txt')

# image_dir = os.path.join(configuration.ROOT_DIR,'image_02') 
image_dir = '/home/SamruddhiPai/Desktop/DeepFusionMOT/datasets/kitti/train/image_02_train'
sequence = ['0000']#os.listdir(image_dir)
for seq in sequence:
    image_files = os.path.join(image_dir,seq) + '/*.png'
    print(image_files)
    images = sorted(glob.glob(image_files))
    for image in images:
        # if '000002' in image: 
        current_frame =  Path(image).name.split('.')[0]
        print('image file',image,'current frame',current_frame)
        a1 = process_image(image, configuration.ROOT_DIR, current_frame, depth_generator, segmentation_model, frustum_convnet, calibration, configuration.SHOW_IMAGE,kitti=True)
                            