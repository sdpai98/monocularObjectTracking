import numpy as np
import cv2
import torch
import urllib
import matplotlib.pyplot as plt

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
#filename = "/home/ubuntu/18744/DepthPerception/saved_copy/carla_images/image_224.jpg"
data_filename = "/home/ubuntu/18744/DepthPerception/saved_copy/calib.npy"
K = np.load(data_filename)
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(filename)
print(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)
    print(prediction.size())
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()


depth = prediction.cpu().numpy()
print(depth)

print(depth.shape)
rows, cols = depth.shape
c, r = np.meshgrid(np.arange(cols), np.arange(rows))
points = np.stack([c, r, depth])
print("points shape")
print(points.shape)
points = points.reshape((3, -1))
points = points.T
f_u = K[0, 0]
f_v = K[1, 1]
c_u =  K[0, 2]
c_v =  K[1, 2]
b_x = 0
b_y = 0
def project_image_to_rect(uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
        y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

pseudo_lidar = project_image_to_rect(points)
print("pseudo-lidar shape")
print(np.shape(pseudo_lidar))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = pseudo_lidar[:, 0]
ys = pseudo_lidar[:, 1]
zs = pseudo_lidar[:, 2]

np.savez("3d_data", xs = xs, ys = ys, zs = zs)
ax.scatter3D(xs, ys, zs)
plt.figure()
plt.subplot(211)
plt.imshow(img)
plt.subplot(212)
plt.imshow(depth)
plt.show()
