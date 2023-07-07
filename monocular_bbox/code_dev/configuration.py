import sys
sys.path.append('/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.10-py3.6-linux-x86_64.egg')
import carla


SCENE_WEATHER = 'morning'

# Town for map load -> town0/7
WORLD_MAP = 'town05' #02

# Path to the CARLA egg file
CARLA_EGG_PATH = "/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.10-py3.6-linux-x86_64.egg"

# Simulation configuration
FPS = 40
NUM_FRAMES = 1000
IGNORE_NUM_FRAMES = 10
NUM_VEHICLES = 300

# Image dimensions
IMWIDTH = 640
IMHEIGHT = 480
IMWIDTH_FRUSTUM = 1242
IMHEIGHT_FRUSTUM = 375
IMFOV = 90

#Data save path
# ROOT_DIR = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/KITTI_mini/object/training'
# ROOT_DIR = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/frustum-convnet/data/kitti/training'
# ROOT_DIR = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/Data/Tracking'
ROOT_DIR = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/Data/Carla'

# Choice of depth model
# DEPTH_MODEL = 'midas'
DEPTH_MODEL = 'packnet'

# PACKNET_CHECKPOINT_FILE = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/pretrained_models/kitti_trained/default_config-overfit_kitti_copy-2023.03.18-23h07m07s/epoch=01_KITTI_tiny-kitti_tiny-velodyne-loss=0.000.ckpt'
PACKNET_CHECKPOINT_FILE = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/PackNet01_MR_selfsup_D.ckpt'

# Flag to display or not display image
SHOW_IMAGE = True

SEGMENTATION_THRESH = 0.7
SEGMENTATION_NETWORK = 'detectron'
pred_classes_mapping = ['background',
 'Pedestrian',
 'Cyclist',
 'Car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush']
