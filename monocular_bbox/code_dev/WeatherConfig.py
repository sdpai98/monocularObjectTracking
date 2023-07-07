import sys
import os
import configuration as cfg
sys.path.append(cfg.CARLA_EGG_PATH)
import carla

class WeatherConfig:
	def __init__(self):
		self.cloudiness = None  # 0.0 to 100.0
		self.precipitation = None  # 0.0 to 100.0
		self.precipitation_deposits = None  # 0.0 to 100.0
		self.wind_intensity = None  # 0.0 to 100.0
		self.sun_azimuth_angle = None  # 0.0 to 360.0
		self.sun_altitude_angle = None  # -90.0 to 90.0

	def get_weather_palette(self):
		return {'morning':self.morning(), 'mid_morning':self.mid_morning(), 
				'afternoon':self.afternoon(), 'twilight':self.twilight()}


	def morning(self):
		self.cloudiness = 20.0
		self.precipitation = 90.0
		self.precipitation_deposits = 30.0
		self.wind_intensity = 30.0
		self.sun_azimuth_angle = 0.0
		self.sun_altitude_angle = 30.0
		return [self.cloudiness, self.precipitation, self.precipitation_deposits, self.wind_intensity,
				self.sun_azimuth_angle, self.sun_altitude_angle]


	def mid_morning(self):
		self.cloudiness = 30.0
		self.precipitation = 0.0
		self.precipitation_deposits = 60.0
		self.wind_intensity = 30.0
		self.sun_azimuth_angle = 00.0
		self.sun_altitude_angle = 80#80.0  # 45
		return [self.cloudiness, self.precipitation, self.precipitation_deposits, self.wind_intensity,
		        self.sun_azimuth_angle, self.sun_altitude_angle]

	def afternoon(self):
		self.cloudiness = 50.0
		self.precipitation = 0.0
		self.precipitation_deposits = 40.0
		self.wind_intensity = 30.0
		self.sun_azimuth_angle = 0.0
		self.sun_altitude_angle = -40.0
		return [self.cloudiness, self.precipitation, self.precipitation_deposits, self.wind_intensity,
		        self.sun_azimuth_angle, self.sun_altitude_angle]

	def twilight(self):
		self.cloudiness = 30.0
		self.precipitation = 30.0
		self.precipitation_deposits = 0.0
		self.wind_intensity = 30.0
		self.sun_azimuth_angle = 0.0
		self.sun_altitude_angle = -60.0
		return [self.cloudiness, self.precipitation, self.precipitation_deposits, self.wind_intensity,
		        self.sun_azimuth_angle, self.sun_altitude_angle]
