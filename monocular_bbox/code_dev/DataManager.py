import os
from datetime import datetime

class DataManager():
	def __init__(self, root_directory):
		self.data_types = {'images': 'images', 'dm': 'depth_maps', 'dgt': 'depth_ground_truth', 
							'cc':'cam_calib', 'pcl': 'point_clouds'}
		date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
		self.capture_dir_name = f"capture_{date}"
		self.capture_path = os.path.join(root_directory, self.capture_dir_name)
		self.image_path = os.path.join(self.capture_path, self.data_types['images'])
		self.depth_path = os.path.join(self.capture_path, self.data_types['dm'])
		self.ground_truth_path = os.path.join(self.capture_path, self.data_types['dgt'])
		self.calib_path = os.path.join(self.capture_path, self.data_types['cc'])
		self.pcl_path = os.path.join(self.capture_path, self.data_types['pcl'])

		os.makedirs(self.capture_path)
		os.makedirs(self.image_path)
		os.makedirs(self.depth_path)
		os.makedirs(self.ground_truth_path)
		os.makedirs(self.calib_path)
		os.makedirs(self.pcl_path)

	def get_capture_path(self):
		return self.capture_path