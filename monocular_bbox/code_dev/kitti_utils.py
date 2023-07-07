import numpy as np

def ParseCalibrationDict(raw_calib):

  calib = {}
  calib['P0'] = raw_calib['P0'].reshape([3, 4])
  calib['P1'] = raw_calib['P1'].reshape([3, 4])
  calib['P2'] = raw_calib['P2'].reshape([3, 4])
  calib['P3'] = raw_calib['P3'].reshape([3, 4])

  # R0_rect contains a 3x3 matrix which you need to extend to a 4x4 matrix by
  # adding a 1 as the bottom-right element and 0's elsewhere.
  extended_r0_rect = np.eye(4)
  extended_r0_rect[:3, :3] = raw_calib['R0_rect'].reshape([3, 3])
  calib['R0_rect'] = extended_r0_rect

  # Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
  # in the same way!
  extended_tr_imu_to_velo = np.eye(4)
  extended_tr_imu_to_velo[:3, :4] = raw_calib['Tr_imu_to_velo'].reshape([3, 4])
  calib['Tr_imu_to_velo'] = extended_tr_imu_to_velo

  extended_tr_velo_to_cam = np.eye(4)
  extended_tr_velo_to_cam[:3, :4] = raw_calib['Tr_velo_to_cam'].reshape([3, 4])
  calib['Tr_velo_to_cam'] = extended_tr_velo_to_cam

  return calib

def LoadCalibrationFile(filepath):

  raw_calib = {}
  f = open(filepath, 'r')
  lines = f.readlines()

  for line in lines:
    line = line.strip()
    if not line:
      continue
    key, value = line.split(':', 1)

    raw_calib[key] = np.array([float(x) for x in value.split()])

  f.close()
  return ParseCalibrationDict(raw_calib)
