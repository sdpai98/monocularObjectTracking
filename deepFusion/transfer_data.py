import os
import sys

def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)

scr_root = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/Data/Carla/capture_2023_06_27-10_32_53_PM'
dest_root = '/home/SamruddhiPai/Desktop/DeepFusionMOT/datasets/kitti/carla-dd3d'
img_folder = os.path.join(dest_root,'image_02_carla-dd3d')
det_2d_folder = os.path.join(dest_root,'2D_rrc_Car_carla')
det_3d_folder = os.path.join(dest_root,'3D_pointrcnn_Car_carla')
calib_folder = os.path.join(dest_root,'calib_carla-dd3d')
seq = '0000'

# create all the folders if they do not exist
create_folder(dest_root)
create_folder(img_folder)
create_folder(det_2d_folder)
create_folder(det_3d_folder)
create_folder(calib_folder)

### images
scr_img_path = os.path.join(scr_root,'images')
dest_img_path = os.path.join(img_folder,seq)
# import pdb; pdb.set_trace()
if not os.path.exists(dest_img_path):
    os.mkdir(dest_img_path)
os.system('mv %s/* %s/.'%(scr_img_path,dest_img_path))

### 2D detections
scr_2d = os.path.join(scr_root,'detections_2d.txt')
dest_2d = os.path.join(det_2d_folder, '%s.txt'%(seq))
cmd_2d = 'cp %s %s'%(scr_2d, dest_2d)
os.system(cmd_2d)

### 3D detections
scr_3d = os.path.join(scr_root,'detections_3d.txt')
dest_3d = os.path.join(det_3d_folder,'%s.txt'%(seq))
cmd_3d = 'cp %s %s'%(scr_3d, dest_3d)
os.system(cmd_3d)

### Calib
# scr_calib = os.path.join(dest_root,'calib_carla/%4d.txt'%(int(seq)-1))
scr_calib = '/home/SamruddhiPai/Desktop/DeepFusionMOT/datasets/kitti/carla/calib_carla/0000.txt'
dest_calib = os.path.join(calib_folder,'%s.txt'%(seq))
cmd_calib = 'cp %s %s'%(scr_calib, dest_calib)
os.system(cmd_calib)