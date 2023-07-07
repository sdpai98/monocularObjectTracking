import sys
import cv2
import os
import queue
import numpy as np
import configuration as config
import random
import data_processing
from WeatherConfig import WeatherConfig
sys.path.append(config.CARLA_EGG_PATH)

import carla 


class SimWorld:
	
	def __init__(self, scene, weather):
		# Set up connection to CARLA server
		self.client = carla.Client('127.0.0.1', 2000)
		self.client.set_timeout(20.0)

		# Load world corresponding to the map 'scene' and set weather
		self.world = self.client.load_world(scene)
		self.weather_options = WeatherConfig().get_weather_palette()
		print("Weather options are:", self.weather_options[weather])
		weather = carla.WeatherParameters(*self.weather_options[weather])
		self.world.set_weather(weather)

		# Create ego vehicle object
		#self.ego_vehicle = self.spawn_ego_vehicle()

		# Notify server to update environment with ego vehicle
		self.world.tick()

		# Lists to hold actors in the scene
		self.vehicles_list = []
		self.sensors_list = []
		self.all_id = []
		self.batch = []
		self.actors = []
		
	'''
		brief: This function setus up the CARLA environment in synchronous mode.
				In this mode, the client controls the simulation rate of the envi-
				ronment
	'''
	def set_synchronous_mode(self, synch_mode = True, fps = 10, no_render = False):
		settings = self.world.get_settings() 		# get current world settings
		settings.synchronous_mode = True 			# Enable synchronous mode
		settings.fixed_delta_seconds = 1.0/fps 		# Set expected simulation rate
		settings.no_rendering_mode = False 			# Enable hi fidelity rendering
		self.world.apply_settings(settings)

	'''
		brief: Spawn the ego vehicle for simulation 
	'''
	def spawn_ego_vehicle(self):
		blueprint_library = self.world.get_blueprint_library()	# get available actor blueprints			
		bp = blueprint_library.filter('model3')[0]				# Extract model of interest
		spawn_point = random.choice(self.world.get_map(			# Pick a spawn point on the map
										).get_spawn_points())

		vehicle = self.world.try_spawn_actor(bp, spawn_point)	# Try to spawn actor at location

		while(vehicle is None):									# Repeat until vehicle is spawned
			spawn_point = random.choice(self.world.get_map().get_spawn_points())
			vehicle = self.world.try_spawn_actor(bp, spawn_point)
		# vehicle.set_autopilot(False)
		print('Ego vehicle spawned......')
		# vehicle.set_attribute('velocity',0)
		self.ego_vehicle = vehicle
		print('ego location', self.ego_vehicle.get_location())
		return vehicle

	'''
		brief: Spawn the remaining actor vehicles
	'''
	def spawn_vehicles(self, num_vehicles):
		# Access CARLA traffic manager
		tm = self.client.get_trafficmanager(8000)	

		# Ensure traffic manager is also in synch mode	
		tm.set_synchronous_mode(True)		

		# Extract vehicles			
		vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')	
		spawn_points = self.world.get_map().get_spawn_points()	# Get spawn points
		num_spawn_points = len(spawn_points)

		# Shuffle spawn points randomly
		if num_vehicles < num_spawn_points:
			random.shuffle(spawn_points)				
		else:
			num_vehicles = num_spawn_points

		# Alias some CARLA functions
		SpawnActor = carla.command.SpawnActor  			
		SetAutopilot = carla.command.SetAutopilot 
		FutureActor = carla.command.FutureActor

		# --------------
		# Spawn vehicles
			## All vehicles are controlled by the CARLA traffic manager's autopilot function
		# --------------
		print('Total number of spawn_points available', len(spawn_points))
		distant_count = 0
		ego_location = self.ego_vehicle.get_location()
		for n, transform in enumerate(spawn_points):
			if n >= num_vehicles:
				break
			# print(transform)
			if ego_location.distance(transform.location) > 150 and ego_location.distance(transform.location) < 1000:
				distant_count+=1
			
				blueprint = random.choice(vehicle_blueprints)
				while int(blueprint.get_attribute('number_of_wheels')) <= 2:
					blueprint = random.choice(vehicle_blueprints)
				# Taking out bicycles and motorcycles, since the semantic/bb labeling for that is mixed with pedestrian
				# if int(blueprint.get_attribute('number_of_wheels')) > 2:
				if blueprint.has_attribute('color'):
					color = random.choice(blueprint.get_attribute('color').recommended_values)
					blueprint.set_attribute('color', color)
				if blueprint.has_attribute('driver_id'):
					driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
					blueprint.set_attribute('driver_id', driver_id)
				blueprint.set_attribute('role_name', 'autopilot')
				self.batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
		print('Distance vehicle found %f times!!!'%(distant_count))
		# Apply commands to actors in a single batch and handle any errors
		for response in self.client.apply_batch_sync(self.batch):
			if response.error:
				print("Error in setting batch sync")
			else:
				self.vehicles_list.append(response.actor_id)
				self.actors.append(response)

		# Tick the server to update the environment and add new vehicles
		self.world.tick()
		# print('Vehicles list', self.vehicles_list)
		return self.vehicles_list

	'''
		brief: Configure the RGB camera sensor
	'''
	def configure_camera(self, vehicle, cam_imwidth, cam_imheight, cam_fov, x_loc, z_loc, queue):

		# get actor blueprint for the RGB camera
		cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')

		# set width, height of image and camera FOV
		cam_bp.set_attribute('image_size_x', f'{cam_imwidth}')
		cam_bp.set_attribute('image_size_y', f'{cam_imheight}')
		cam_bp.set_attribute('fov', f'{cam_fov}')

		# The camera is attached to some location related to the vehicle centre
		transform = carla.Transform(carla.Location(x=x_loc, z=z_loc))

		# spawn the camera in the scene and set some paraeters
		self.rgb_camera = self.world.spawn_actor(cam_bp, transform, attach_to = vehicle)
		self.rgb_camera.blur_amount = 0.0
		self.rgb_camera.motion_blur_intensity = 0
		self.rgb_camera.motion_max_distortion = 0

		# construct camera calibration matrix from the camera attributes
		calibration = np.zeros((3,4))
		calibration[0, 2] = cam_imwidth / 2.0
		calibration[1, 2] = cam_imheight / 2.0
		calibration[0, 0] = calibration[1, 1] = cam_imwidth / (2.0 * np.tan(cam_fov * np.pi / 360.0))
		calibration[2, 2] = 1.0
		self.rgb_camera.calibration = calibration

		# setup a listener callback function for the camera - we simply enque images
		self.rgb_camera.listen(queue.put)
		
		# add camera to sensors list - to be destroyed when simulation terminates
		self.sensors_list.append(self.rgb_camera)


		return self.rgb_camera


	'''
		brief: Configure depth sensor to capture ground truth
	'''
	def configure_depth_sensor(self, vehicle, cam_imwidth, cam_imheight, cam_fov, x_loc, z_loc, queue):

		# get blueprint and setup camera FOV
		depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
		depth_bp.set_attribute('image_size_x', f'{cam_imwidth}')
		depth_bp.set_attribute('image_size_y', f'{cam_imheight}')
		depth_bp.set_attribute('fov', f'{cam_fov}')

		# spawn the depth sensor and set up listener call back functions
		spawn_point = carla.Transform(carla.Location(x=x_loc, z=z_loc))
		self.depth_camera = self.world.spawn_actor(depth_bp, spawn_point, attach_to=vehicle)
		self.depth_camera.listen(queue.put)
		self.sensors_list.append(self.depth_camera)

		return self.depth_camera

	'''
		brief: primary data acquisition function
	'''
	def acquire_data(self, imwidth, imheight, imfov, frame_rate, num_frames, depth_model, \
		segmentation_model, frustum_convnet, dd3d_model, data_file_path, show_image, sample_rate = 1):

		current_frame = 0 

		# Set up queue objects to enque images and depth maps
		camera_queue = queue.Queue()
		depth_queue = queue.Queue()

		
		# configure RGB camera
		rgb_cam = self.configure_camera(self.ego_vehicle, imwidth, imheight, imfov, 2.5, 2.7, camera_queue)

		# configure depth sensor at same location as camera
		self.configure_depth_sensor(self.ego_vehicle, imwidth, imheight, imfov, 2.5, 2.7, depth_queue)
		
		# save camera calibration data and model info in given path
		params_dict = {}
		params_dict['depth_model'] = config.DEPTH_MODEL
		params_dict['condition'] = config.SCENE_WEATHER
		file_name = 'cam_calib/calib'
		file_n = 'cam_calib/model'
		file_path = os.path.join(data_file_path, file_name)
		file_path1 = os.path.join(data_file_path, file_n)
		np.save(file_path, rgb_cam.calibration)
		np.save(file_path1, params_dict)

		# Now run simulation for given number of frames
		while (current_frame< num_frames):
			

			print(current_frame)
			current_frame += 1

			# advance one time step in the simulation
			self.world.tick()

			# when data is available from camera
			if(not camera_queue.empty()):
				if(current_frame <= config.IGNORE_NUM_FRAMES):
					discard_image = camera_queue.get()
					discard_depth = depth_queue.get()
				elif current_frame%sample_rate != 0:
					discard_image = camera_queue.get()
					discard_depth = depth_queue.get()                          
					print('Skipping %d frame'%(current_frame))
				else:
					# apply vehicle physics update to all actors
					self.client.apply_batch([carla.command.SetAutopilot(x, True) for x in [v for v in self.world.get_actors().filter("vehicle.*")]])
					# self.ego_vehicle.set_autopilot(True)
					# all_actors = [v for v in self.world.get_actors().filter("vehicle.*")]
					# print('Actor velocities!!!!!!!!')
					# actors_velocity = [print(x.get_velocity()) for x in all_actors]
					# print([x() for x in actors_velocity])

					# process the camera image and depth image
					# import pdb; pdb.set_trace()
					# a1 = data_processing.process_image(camera_queue.get(), data_file_path, current_frame, depth_model, segmentation_model, frustum_convnet, self.rgb_camera.calibration, show_image)
					# b1 = data_processing.process_depth(depth_queue.get(), data_file_path, current_frame, depth_model, self.rgb_camera.calibration, show_image, carla.ColorConverter.Raw)

					a1 = data_processing.process_image_dd3d(camera_queue.get(),  data_file_path, current_frame, self.rgb_camera.calibration, dd3d_model, segmentation_model)