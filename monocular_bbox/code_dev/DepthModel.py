import numpy as np
from PIL import Image
import configuration as config
import torch
import sys
import cv2 as cv2

sys.path.append("/home/SamruddhiPai/Desktop/monocular_3d_boundingBox/code_dev/packnet-sfm")
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor

class DepthModel:
	def __init__(self, model):
		self.transform = None
		self.device = None
		self.config = None
		self.state_dict = None
		if(model == 'midas'):
			self.model_name = 'midas'
		elif(model == 'packnet'):
			self.model_name = 'packnet'
		else:
			self.model_name = 'midas'

	def configure_model(self, checkpoint_file = config.PACKNET_CHECKPOINT_FILE):
		if(self.model_name == 'midas'):
			model_type = "DPT_Large"
			midas = torch.hub.load("intel-isl/MiDaS", model_type)
			self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
			midas.to(self.device)
			midas.eval()
			midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
			if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
				transform = midas_transforms.dpt_transform
			else:
				transform = midas_transforms.small_transform
			self.transform = transform
			self.model = midas
		elif(self.model_name == 'packnet'):
			hvd_init()
			self.config_1, self.state_dict = parse_test_file(checkpoint_file)
			model_wrapper = ModelWrapper(self.config_1, load_datasets=False)
			model_wrapper.load_state_dict(self.state_dict)
			if torch.cuda.is_available():
				model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=None)
			model_wrapper.eval()
			self.model = model_wrapper
			return None
		else:
			return None

	def generate_depth_map(self, img):
		if(self.model_name == 'midas'):

			input_batch = self.transform(img).to(self.device)
			with torch.no_grad():
				prediction = self.model(input_batch)
				prediction = torch.nn.functional.interpolate(
				        prediction.unsqueeze(1),
				        size=img.shape[:2],
				        mode="bicubic",
				        align_corners=False,
				    ).squeeze()	
			depth = prediction.cpu().numpy()
			return depth, None
		elif(self.model_name == 'packnet'):
			return self.infer_and_save_depth(img, None, self.model, (config.IMHEIGHT, config.IMWIDTH), False, None)
		else:
			return None


	@torch.no_grad()
	def infer_and_save_depth(self, image, output_file, model_wrapper, image_shape, half, save):
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
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(np.uint8(image)).convert('RGB')
		# Resize and to tensor
		image = resize_image(image, image_shape)
		image = to_tensor(image).unsqueeze(0)

		# Send image to GPU if available
		if torch.cuda.is_available():
		    image = image.to('cuda:{}'.format(rank()), dtype=None)

		# Depth inference (returns predicted inverse depth)
		pred_inv_depth = model_wrapper.depth(image)['inv_depths'][0]
		depth_npy = inv2depth(pred_inv_depth)
		# print(type(depth_npy))
		if save == 'npz' or save == 'png' or save == 'npy':
			pass
		    # Get depth from predicted depth map and save to different formats
		    # filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
		    # print('Saving {} to {}'.format(
		    #     pcolor(input_file, 'cyan', attrs=['bold']),
		    #     pcolor(filename, 'magenta', attrs=['bold'])))
		    # depth_npy = inv2depth(pred_inv_depth)
		    #write_depth(filename, depth=inv2depth(pred_inv_depth))
		    #return depth_npy
		else:
		    # Prepare RGB image
		    rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
		    # Prepare inverse depth
		    viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
		    # Concatenate both vertically
		    image = np.concatenate([rgb, viz_pred_inv_depth], 0)
		    # Save visualization
		    # print('Saving {} to {}'.format(
		    #     pcolor(input_file, 'cyan', attrs=['bold']),
		    #     pcolor(output_file, 'magenta', attrs=['bold'])))
		    # imwrite(output_file, image[:, :, ::-1])
		    #return image[:, :, ::-1]
		    depth_npy = depth_npy.detach().squeeze().cpu().numpy()
		    return depth_npy, viz_pred_inv_depth

