# Image dimensions
IMWIDTH = 1242
IMHEIGHT = 375
IMFOV = 90

#Data save path
ROOT_DIR = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/KITTI_mini/object/testing'

# Choice of depth model
DEPTH_MODEL = 'packnet'

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