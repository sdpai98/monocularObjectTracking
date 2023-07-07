# import numpy as np
# import cv2
# import torch
# from kitti_utils import *
# import matplotlib.pyplot as plt

# IMAGE_FILE_PATH = '/home/ubuntu/18744/Data/KITTI_mini/object/testing/image_2/000008.png'
# CALIB_FILE_PATH = '/home/ubuntu/18744/Data/KITTI_mini/object/testing/calib/000008.txt'

# model_type = "DPT_Large"

# img = cv2.imread(IMAGE_FILE_PATH)
# calib_dict = LoadCalibrationFile(CALIB_FILE_PATH)

# K = calib_dict['P2']

# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform

# input_batch = transform(img).to(device)
# with torch.no_grad():
#     prediction = midas(input_batch)
#     print(prediction.size())
#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()


# depth = prediction.cpu().numpy()


# print(depth.shape)
# rows, cols = depth.shape
# c, r = np.meshgrid(np.arange(cols), np.arange(rows))
# points = np.stack([c, r, depth])
# print("points shape")
# print(points.shape)
# points = points.reshape((3, -1))
# points = points.T
# f_u = K[0, 0]
# f_v = K[1, 1]
# c_u =  K[0, 2]
# c_v =  K[1, 2]
# b_x = 0
# b_y = 0
# def project_image_to_rect(uv_depth):
#         ''' Input: nx3 first two channels are uv, 3rd channel
#                    is depth in rect camera coord.
#             Output: nx3 points in rect camera coord.
#         '''
#         n = uv_depth.shape[0]
#         x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
#         y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
#         pts_3d_rect = np.zeros((n, 3))
#         pts_3d_rect[:, 0] = x
#         pts_3d_rect[:, 1] = y
#         pts_3d_rect[:, 2] = uv_depth[:, 2]
#         return pts_3d_rect

# pseudo_lidar = project_image_to_rect(points)
# print("pseudo-lidar shape")
# print(np.shape(pseudo_lidar))
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# xs = pseudo_lidar[:, 0]
# ys = pseudo_lidar[:, 1]
# zs = pseudo_lidar[:, 2]

# np.savez("3d_data", xs = xs, ys = ys, zs = zs)
# # ax.scatter3D(xs, ys, zs)
# # plt.figure()
# # plt.subplot(211)
# # plt.imshow(img)
# # plt.subplot(212)
# # plt.imshow(depth)
# # plt.show()

import cv2
import torch
from DepthModel import *
import open3d as o3d 
import numpy as numpy
from kitti_utils import *
import matplotlib.pyplot as plt


IMAGE_FILE_PATH = '/home/ubuntu/Downloads/000000.jpeg'
CALIB_FILE_PATH = '/home/ubuntu/Downloads/kitti/testing/calib/000000.txt'

raw_img = o3d.io.read_image(IMAGE_FILE_PATH) #cv2.imread(IMAGE_FILE_PATH)
img = np.array(raw_img)
# calib_dict = LoadCalibrationFile(CALIB_FILE_PATH)

# K = calib_dict['P2']

#K = np.loadtxt(CALIB_FILE_PATH)
K = np.array([320.00000000000006, 0.0, 320.0, 0.0, 0.0, 320.00000000000006, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0])
K = np.reshape(K, (3, 4))
f_u = K[0, 0]
f_v = K[1, 1]
c_u =  K[0, 2]
c_v =  K[1, 2]

# model_type = "DPT_Large"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform

# input_batch = transform(img).to(device)
# with torch.no_grad():
#     prediction = midas(input_batch)
#     print(prediction.size())
#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()

model = DepthModel('packnet')
model.configure_model()
prediction = model.generate_depth_map(img)
cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(img.shape[1], img.shape[0], f_u, f_v, c_u, c_v)
depth = prediction[0]
depth_raw = o3d.cuda.pybind.geometry.Image(depth)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(raw_img, depth_raw)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, cam_intrinsics)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

print(type(pcd))
o3d.visualization.draw_geometries([pcd])

plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()
