import os
import sys
import pdb
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.cm import get_cmap
import configuration as config
import torch
import time
from Segmentation import Segmentation
# from cv2 import cv2
import cv2
import copy
from PIL import Image
from pathlib import Path
sys.path.append('/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.10-py3.6-linux-x86_64.egg')
import carla
import torchvision as tv

sys.path.append("/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/packnet-sfm")
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor

sys.path.append("/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/frustum-convnet")
import kitti.kitti_util as kitti_util
from kitti.draw_util import show_image_with_boxes, show_image_with_boxes3d
# from visualize_3dbb import *
from configs_1.config import cfg

sys.path.append("/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev")
import bounding_box_optimization as bbo

### imports from dd3d ###
from pyquaternion import Quaternion
from pytorch3d.transforms import transform3d as t3d
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, quaternion_to_axis_angle, matrix_to_axis_angle
# from pytorch3d.transforms import matrix_to_axis_angle

computation_time_list = []
time_3d = 0
total_frames = 0
avg_time_dd3d = 0

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

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
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    pseudo_cloud_rect = calib.project_image_to_rect(points)
    cloud = calib.project_rect_to_velo(pseudo_cloud_rect)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return pseudo_cloud_rect, cloud[valid]

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
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    # Resize and to tensor
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
        return image[:, :, ::-1]

def project_image_to_rect(uv_depth, calibration):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - calibration[0, 2]) * uv_depth[:, 2]) / calibration[0, 0] + calibration[0, 3]
    y = ((uv_depth[:, 1] - calibration[1, 2]) * uv_depth[:, 2]) / calibration[1, 1] + calibration[1, 3]
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect

def generate_point_cloud(depth, camera_matrix):
    # print(np.shape(depth))
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    pseudo_cloud_rect = project_image_to_rect(points, camera_matrix)
    return pseudo_cloud_rect

def draw_rectangle(img, boxes):
    rmask_rcnn_image = copy.deepcopy(img)
    for i in boxes:
        cv2.rectangle(rmask_rcnn_image,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(0,255,0),1)
    
    return rmask_rcnn_image

def meshgrid_points(depth):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    return points

def process_depth_map(depth_map):
    disp_map = (depth_map).astype(np.float32)#/256.
    pseudo_lidar_in_img_fov = meshgrid_points(disp_map)
    print("pseudo_lidar_in_img_fov", pseudo_lidar_in_img_fov.shape)
    pseudo_lidar_in_img_fov = np.concatenate([pseudo_lidar_in_img_fov, \
                            np.ones((pseudo_lidar_in_img_fov.shape[0], 1))], 1)
    pseudo_lidar_in_img_fov = pseudo_lidar_in_img_fov.astype(np.float32)
    print("pseudo_lidar_in_img_fov", pseudo_lidar_in_img_fov.shape)
    return pseudo_lidar_in_img_fov

def window_cv2(win_name, img):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)

def save_cv2(path,img):
    cv2.imwrite(path,img)

def process_image(image, folder, frame_id, depth_model, seg_model, frustum_convnet, calib, show_image = False, kitti = False):
    global time_3d
    global total_frames
    calibration = {}
    process_image_dd3d(image,calib)
    if not isinstance(calib,dict):
        calibration['P2'] = calib
    else:
        calibration = calib
    # print("Calibration matrix")
    # print(calibration)
    if kitti:
        result_dir = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/output/tracking'
        i3 = cv2.imread(image)
        # i = cv2.resize(i, (config.IMHEIGHT, config.IMWIDTH))
        # i3 = i.reshape((config.IMHEIGHT, config.IMWIDTH, 3))
        print('Image loaded',i3.shape)
        # depth_meters = process_depth_kitti(depth_map, folder+'/depth_in_meters', frame_id)

    else:
        result_dir = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/output/carla'
        i = np.array(image.raw_data)
        i2 = i.reshape((config.IMHEIGHT, config.IMWIDTH, 4))
        i3 = i2[:, :, :3]
        image_file = 'images/{}.png'.format(str(frame_id))
        image_path = os.path.join(folder, image_file)
        # print('!!!!!!!Saving image!!!!! at',image_path) 
        save_cv2(image_path,i3)
        depth_meters = process_depth(image, folder+'/depth_ground_truth', frame_id, depth_model, calibration, show_image, carla.ColorConverter.Raw)
  
    st = time.time()
    depth_map, depth_viz = depth_model.generate_depth_map(i3)
    end = time.time()
    print('Time taken for generating depth map ',end-st)
    if kitti:
        x_shape,y_shape = i3.shape[:2]
        depth_map = np.resize(depth_map,(x_shape,y_shape))
        print('Depth map shape',depth_map.shape)
    else:
        depth_map_meters = postprocess_depth(depth_map, folder+'/depth_maps', frame_id) #, depth_model, calibration, show_image, carla.ColorConverter.Raw)

    # print("depth_map_r Min Max ", np.min(depth_map_r), np.max(depth_map_r))
    # print("depth_map_meters Min Max ", np.min(depth_map_meters), np.max(depth_map_meters))
    st = time.time()
    num_instances, classes, boxes, scores = seg_model.get_bounding_boxes(i3)
    end = time.time()
    print('Time taken for 2D detection ',end-st)
    draw_img = draw_rectangle(i3, boxes)

    pseudo_lidar_in_img_fov = process_depth_map(depth_map)

    st = time.time()
    frustum_convnet.prepare_data(num_instances, classes, boxes, scores, \
        'saved_pickle_file', calibration, pseudo_lidar_in_img_fov)
    end = time.time()
    print('Time taken for data prep ',end-st)
    
    st = time.time()
    convnet_bbox = frustum_convnet.test(frame_id=int(frame_id),folder=folder)
    end = time.time()
    print('Time taken for 3D detections ',end-st)
    time_3d+=(end-st)
    total_frames += 1
    #image_aapli = cv2.imread("/home/ubuntu/18744/manucular_vision/DepthPerception/code_dev/frustum-convnet/data/kitti/testing/image_2/000000.png")
    if cfg.TEST.METHOD == 'nms':
        # print('In nms loop!!!')
        det_results_final,bboxes_3d = frustum_convnet.write_detection_results_nms(folder, convnet_bbox, threshold=cfg.TEST.THRESH, frame_id=int(frame_id),kitti=True)
    else:
        det_results_final,bboxes_3d = frustum_convnet.write_detection_results(folder, convnet_bbox, frame_id=int(frame_id),kitti=True)
    # print(type(image_aapli))
    # print(type(i3))
    # img_3dbb = viz_3dbb(i3, convnet_bbox, boxes)
    _ , img_3dbb, new_2d = show_image_with_boxes(i3, bboxes_3d, calibration,show3d=False,show=False)
    projected_2d_boxes = [(int(obj.xmin), int(obj.ymin), int(obj.xmax), int(obj.ymax)) \
                            for obj in bboxes_3d]
    img_pred_2dbox = draw_rectangle(i3, projected_2d_boxes)
    # print("_______________________________________")
    # print('!!!!!Projected 2d boxes!!!!!!!',projected_2d_boxes)
    # print('!!!!!New 2d boxes!!!!!!!',new_2d)
    # print('!!!!!! actual 2D boxes !!!!!!', boxes)
    # print("_______________________________________")
    # Preparing data for frustum convnet

    # Save the depth map
    # depth_map_file = 'depth_maps/depth_map_{}'.format(str(frame_id))
    # depth_path = os.path.join(folder, depth_map_file)
    # np.save(depth_path, depth_map)

    end_time = time.time()
    computation_time_list.append(end_time - st)
    average_inference_time = np.mean(computation_time_list)
    print("Running average inference time = ", average_inference_time)
    # disp_map = (depth_map).astype(np.float32)/256

    '''
    ToDo: Add frustum convnet code segment here
            - packnet image resizing needs to be looked into
    '''
    # point_cloud = generate_point_cloud(disp_map, calibration)
    # #print(point_cloud)
    if(config.DEPTH_MODEL == 'packnet'):
        depth_viz= ((depth_viz/255) * 255).astype('uint8')
        depth_map_img = cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR)
        depth_map_img_gray = cv2.applyColorMap(depth_map_img, cv2.COLORMAP_MAGMA)
    elif(config.DEPTH_MODEL == 'midas'):
        depth_map_img = np.dstack((depth_map, depth_map, depth_map))
        depth_map_img = ((depth_map_img/255) * 255).astype('uint8')
        depth_map_img_gray = cv2.applyColorMap(depth_map_img, cv2.COLORMAP_HSV)
    else:
        pass
    
    save_boxes_dir = folder+'/bboxes_images'
    if not os.path.isdir(save_boxes_dir):
        os.mkdir(save_boxes_dir)
    save_cv2(save_boxes_dir +'/box2d_%s.png'%(str(frame_id)),draw_img)
    # save_cv2(folder+'_'+str(frame_id)+'_2d_box_projected.png',img_pred_2dbox)
    save_cv2(save_boxes_dir+'/box3d_%s.png'%(str(frame_id)),img_3dbb)
    # save_cv2(folder+'_'+str(frame_id)+'_pred_depth.png',depth_map_img_gray)

    if(show_image == True):
        # pass
        # cv2.imshow("Original image", draw_img)

        window_cv2("2D Bounding Box image", draw_img)
        window_cv2("Depth Map", depth_map_img_gray)
        window_cv2("3D Bounding Box image", img_3dbb)
        window_cv2("Projected 2D Bounding Box image", img_pred_2dbox)
        cv2.waitKey(5)

    # if(folder is not None):
    #     depth_file = 'depth_map_{}.jpg'.format(str(frame_id))
    #     lidar_file = 'point_clouds/pseudo_lidar_{}'.format(str(frame_id))
    #     file_name = 'images/image_{}.jpg'.format(str(frame_id))
    #     file = os.path.join(folder, file_name)
    #     lidar_path = os.path.join(folder, lidar_file)
    #     depth_path = os.path.join(folder, depth_file)
    #     np.save(lidar_path, point_cloud)
    #     cv2.imwrite(file, i3)
    #     #cv2.imwrite(depth_path, depth_map_img_gray)
    print('Average time taken for 3D detection', time_3d/total_frames,time_3d,total_frames)
    return depth_map


def process_depth(image, folder, frame_id, depth_model, calibration, show_image, cmap):
    data = np.array(image.raw_data)
    data = data.reshape((config.IMHEIGHT, config.IMWIDTH, 4))
    data = data.astype(np.float32)
    

    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth_meters = normalized_depth * 1000
    depth_file = 'depth_gt_{}'.format(str(frame_id))
    depth_path = os.path.join(folder, depth_file)
    # print('!!!!!!!Saving depth!!!!! at',depth_path)
    # image.save_to_disk(depth_path,  cmap)
    np.save(depth_path, depth_meters)
    
    return depth_meters


def process_image_kitti(image, folder, frame_id, depth_model, seg_model, frustum_convnet, calibration, show_image = False):
    result_dir = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/output/tracking'
    # result_dir = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/output'
    # print("Calibration matrix")
    # print(calibration)
    i = cv2.imread(image)
    # i = cv2.resize(i, (config.IMHEIGHT, config.IMWIDTH))
    # i2 = i.reshape((config.IMHEIGHT, config.IMWIDTH, 3))
    print('Image loaded',i.shape)
    i3 = i #[:, :, :3]
    st = time.time()
    # cv2.imwrite(Path(image).name,i3)
    depth_map, depth_viz = depth_model.generate_depth_map(i3)

    print("depth_map _________________ ", depth_map.shape)
    print("depth_map Min Max ", np.min(depth_map), np.max(depth_map))
    depth_map = np.resize(depth_map,(1242,375))
    print("depth_map _________________ ", depth_map.shape)

    # ratio = np.max(depth_meters) / np.max(depth_map)
    # depth_map = depth_map * ratio
    # print("depth_map Min Max ", np.min(depth_map), np.max(depth_map))
    # depth_meters = process_depth_kitti(depth_map, folder+'/depth_in_meters', frame_id)
    # print("depth_meters _________________ ", depth_meters.shape)
    # print("depth_meters Min Max ", np.min(depth_meters), np.max(depth_meters))
    # depth_meters = depth_meters / np.max(depth_meters)
    # depth_map_meters = postprocess_depth(depth_map, folder+'/depth_maps', frame_id)
    num_instances, classes, boxes, scores = seg_model.get_bounding_boxes(i3)
    draw_img = draw_rectangle(i3, boxes)

    print('After segmentation!!!!!!!')
    print('num_instances',num_instances)
    print('classes',classes)
    # print('boxes',boxes)
    # print('scores',scores)

    pseudo_lidar_in_img_fov = process_depth_map(depth_map)

    frustum_convnet.prepare_data(num_instances, classes, boxes, scores, \
        'saved_pickle_file', calibration, pseudo_lidar_in_img_fov)
    
    convnet_bbox = frustum_convnet.test()
    # image_aapli = cv2.imread("/home/ubuntu/18744/manucular_vision/DepthPerception/code_dev/frustum-convnet/data/kitti/testing/image_2/000000.png")
    if cfg.TEST.METHOD == 'nms':
        print('In nms loop!!!')
        det_results_final,bboxes_3d = frustum_convnet.write_detection_results_nms(folder, convnet_bbox, threshold=cfg.TEST.THRESH, frame_id=int(frame_id),kitti=True)
    else:
        det_results_final,bboxes_3d = frustum_convnet.write_detection_results(folder, convnet_bbox, frame_id=int(frame_id))
    print("_______________________________________")
    # print(type(image_aapli))
    # print(type(i3))
    print("_______________________________________")
    # img_3dbb = viz_3dbb(i3, convnet_bbox, boxes)
    _,img_3dbb,_ = show_image_with_boxes(i3, bboxes_3d, calibration,show3d=False,show=False)
    print('3d boxes',bboxes_3d)
    # Preparing data for frustum convnet


    # Save the depth map
    # depth_map_file = 'depth_maps/depth_map_{}'.format(str(frame_id))
    # depth_path = os.path.join(folder, depth_map_file)
    # np.save(depth_path, depth_map)

    end_time = time.time()
    computation_time_list.append(end_time - st)
    average_inference_time = np.mean(computation_time_list)
    print("Running average inference time = ", average_inference_time)
    # disp_map = (depth_map).astype(np.float32)/256

    '''
    ToDo: Add frustum convnet code segment here
            - packnet image resizing needs to be looked into
    '''
    # point_cloud = generate_point_cloud(disp_map, calibration)
    # #print(point_cloud)
    if(config.DEPTH_MODEL == 'packnet'):
        depth_viz = ((depth_viz/255) * 255).astype('uint8')
        depth_map_img = cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR)
        depth_map_img_gray = cv2.applyColorMap(depth_map_img, cv2.COLORMAP_MAGMA)
    elif(config.DEPTH_MODEL == 'midas'):
        depth_map_img = np.dstack((depth_map, depth_map, depth_map))
        depth_map_img = ((depth_map_img/255) * 255).astype('uint8')
        depth_map_img_gray = cv2.applyColorMap(depth_map_img, cv2.COLORMAP_HSV)
    else:
        pass

    save_cv2(folder+'_'+frame_id+'_2d_box.png',draw_img)
    save_cv2(folder+'_'+frame_id+'_3d_box.png',img_3dbb)
    # save_cv2(folder+'_'+frame_id+'_pred_depth.png',depth_map_img_gray)
    if(show_image == True):
        # pass
        # cv2.imshow("Original image", draw_img)

        window_cv2("2D Bounding Box image", draw_img)
        window_cv2("Depth Map", depth_map_img_gray)
        window_cv2("3D Bounding Box image", img_3dbb)
        cv2.waitKey(5)

    # if(folder is not None):
    #     depth_file = 'depth_map_{}.jpg'.format(str(frame_id))
    #     lidar_file = 'point_clouds/pseudo_lidar_{}'.format(str(frame_id))
    #     file_name = 'images/image_{}.jpg'.format(str(frame_id))
    #     file = os.path.join(folder, file_name)
    #     lidar_path = os.path.join(folder, lidar_file)
    #     depth_path = os.path.join(folder, depth_file)
    #     np.save(lidar_path, point_cloud)
    #     cv2.imwrite(file, i3)
    #     #cv2.imwrite(depth_path, depth_map_img_gray)
    return depth_map

def process_depth_kitti(image, folder, frame_id):
    data = image
    print('Depth size',data.shape)
    data = data.reshape((config.IMHEIGHT, config.IMWIDTH, 3))
    data = data.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth_meters = normalized_depth * 1000
    depth_file = 'depth_{}'.format(str(frame_id))
    depth_path = os.path.join(folder, depth_file)
    # image.save_to_disk(depth_path,  cmap)
    if not os.path.exists(depth_path):
		    os.makedirs(depth_path)
    np.save(depth_path, depth_meters)
    
    return depth_meters

def postprocess_depth(image, folder, frame_id):
    data = image
    print('Depth size',data.shape)
    data = data.reshape((config.IMHEIGHT, config.IMWIDTH))
    data = data.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    # normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
    # normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    # depth_meters = normalized_depth * 1000
    depth_file = 'depth_{}'.format(str(frame_id))
    depth_path = os.path.join(folder, depth_file)
    # image.save_to_disk(depth_path,  cmap)
    if not os.path.exists(depth_path):
		    os.makedirs(depth_path)
    np.save(depth_path, data)
    
    return data

def process_image_dd3d(image,folder,frame_id,calib,dd3d_model,seg_model):
    global avg_time_dd3d
    global total_frames
    st = time.time()
    i = np.array(image.raw_data)
    i2 = i.reshape((config.IMHEIGHT, config.IMWIDTH, 4))
    i3 = i2[:, :, :3]

    pdb.set_trace()
    ################# NuScenes ######################
    nusc_img_path = '/home/SamruddhiPai/Desktop/dd3d/data/datasets/nuscenes/nuScenes/samples/CAM_FRONT'
    nusc_images = os.listdir(nusc_img_path)
    calib_3x3 = np.array([[1266.417203046554,0.0,816.2670197447984],[0.0,1266.417203046554,491.50706579294757],[0.0,0.0,1.0]])

    for i in nusc_images:
        img_path = os.path.join(nusc_img_path,i)
        i3  = cv2.imread(img_path)

    #################################################
        i4 = torch.from_numpy(i3)
        i4 = i4.permute((2, 0, 1))
        calib_3x3 = torch.from_numpy(calib[:,:3]).float()
        input_dict = {
            'image': i4,
            'intrinsics':calib_3x3
        }
        # pdb.set_trace()
        output = dd3d_model([input_dict])
        # print('output!!!!!!!!', output)
        end = time.time()
        # print('Time taken cceceto process each frame!!', end-st)
        avg_time_dd3d += (end-st)
        total_frames += 1
        if total_frames % 10 == 0:
            time_dd3d = avg_time_dd3d/total_frames
            print('averae time for %d frames is %f'%(total_frames, time_dd3d))
        output_fields = output[0]['instances']._fields
        bboxes_2d = output_fields['pred_boxes'].tensor.detach().cpu().numpy()
        scores_2d = output_fields['scores']
        final_2dbbox_idx = tv.ops.nms(output_fields['pred_boxes'].tensor,scores_2d,0.6)
        final_2dbbox_idx = final_2dbbox_idx.detach().cpu().numpy()
        scores_2d = scores_2d[final_2dbbox_idx]
        bboxes_2d = bboxes_2d[final_2dbbox_idx,:]
        bboxes_3d_scores = output_fields['scores_3d']
        bboxes_3d = output_fields['pred_boxes3d']
        xyz = bboxes_3d.tvec.detach().cpu().numpy()
        hwl = bboxes_3d.size.detach().cpu().numpy()
        depth = bboxes_3d.depth.detach().cpu().numpy()
        classes = output_fields['pred_classes'].detach().cpu().numpy()
        corners_3d = bboxes_3d.corners.detach().cpu().numpy()
        projected_3d = np.array([kitti_util.project_to_image(pts_3d, calib) for pts_3d in corners_3d])
        # num_instances, classes, boxes, scores = seg_model.get_bounding_boxes(i3)
        # draw_img = draw_rectangle(i3, boxes)
        # window_cv2("2D Bounding Box image", draw_img)
        # print('DD3D 2D bbox',bbox_2d)
        # print('Detrectron 2D bbox', boxes)
        # pdb.set_trace()
        mbrs_3d_cpu = [bbo.mbr(pts) for pts in projected_3d]

        if len(mbrs_3d_cpu) != 0:

            image_file = 'images/{}.png'.format(str(frame_id))
            image_path = os.path.join(folder, image_file)
            # print('!!!!!!!Saving image!!!!! at',image_path) 
            save_cv2(image_path,i3)

            mbrs_3d = torch.stack(mbrs_3d_cpu).cuda()
            final_3dbbox_idx = tv.ops.nms(mbrs_3d,bboxes_3d_scores,0.6)
            final_3dbbox_idx = final_3dbbox_idx.detach().cpu().numpy()
            projected_3d = projected_3d[final_3dbbox_idx,:]
            img_pred_2dbox = draw_rectangle(i3, bboxes_2d)
            img_3dbb = show_image_with_boxes3d(i3, projected_3d, calib)
            save_boxes_dir = folder+'/bboxes_images'
            # print('Saving images at!!!!!!!!', save_boxes_dir)
            if not os.path.isdir(save_boxes_dir):
                os.mkdir(save_boxes_dir)
            # save_boxes_dir = '/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/output/carla/dd3d'
            save_cv2(save_boxes_dir +'/box2d_%s.png'%(i),img_pred_2dbox)
            # save_cv2(save_boxes_dir +'/box2d_%s.png'%(str(frame_id)),img_pred_2dbox)
            # save_cv2(folder+'_'+str(frame_id)+'_2d_box_projected.png',img_pred_2dbox)
            save_cv2(save_boxes_dir+'/box3d_%s.png'%(i),img_3dbb)
            # save_cv2(save_boxes_dir+'/box3d_%s.png'%(str(frame_id)),img_3dbb)
            # window_cv2("Projected 2D Bounding Box image", img_pred_2dbox)
            # window_cv2("3D Bounding Box image", img_3dbb)
            # cv2.waitKey(5)
            translation = t3d.Translate(bboxes_3d.tvec)
            R = quaternion_to_axis_angle(bboxes_3d.quat)
            output_det_2d = os.path.join(folder,'detections_2d.txt')
            output_det_3d = os.path.join(folder,'detections_3d.txt')

            # pdb.set_trace()

            with open(output_det_2d,'a') as f:
                for i in range(len(bboxes_2d)):
                    f.write('%d,%f,%f,%f,%f,%f\n'%(frame_id,bboxes_2d[i][0],bboxes_2d[i][1],
                    bboxes_2d[i][2], bboxes_2d[i][3], scores_2d[i]))

            with open(output_det_3d,'a') as f:
                for i in final_3dbbox_idx:
                    f.write('%d,2,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,0\n'%(frame_id,mbrs_3d_cpu[i][0],mbrs_3d_cpu[i][1], mbrs_3d_cpu[i][2], mbrs_3d_cpu[i][3], bboxes_3d_scores[i], hwl[i][0], hwl[i][1], hwl[i][2], xyz[i][0], xyz[i][1], xyz[i][2], R[i][1]))

            # R = quaternion_to_matrix(bboxes_3d.quat)
            # R_mat = matrix_to_axis_angle(quaternion_to_matrix(bboxes_3d.quat))
            # rotation = t3d.Rotate(R=R.transpose(1, 2))
            