# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:00:51 2020

@author: MangalDeep
"""

from scipy.io import loadmat
import pathlib as path


GPS_Data = loadmat('G:/DataSets/Victoria_park/original_MATLAB_dataset/aa3_gpsx.mat')
Laser_Data = loadmat('G:/DataSets/Victoria_park/original_MATLAB_dataset/aa3_lsr2.mat')
Odo_Data = loadmat('G:/DataSets/Victoria_park/original_MATLAB_dataset/aa3_dr.mat')

#Car Param
a  = 0.95 #meters
b  = 0.5  #meters
H  = 0.76 #m meters
L  = 2.83 #meters

Lat_m = GPS_Data['La_m']
Long_m = GPS_Data['Lo_m']
#GPS_T_ms =GPS_Data['timeGPS']

Laser_T_ms = Laser_Data['TLsr']
Laser_dat = Laser_Data['LASER']

Speed_mps = Odo_Data['speed']
Steer_rad = Odo_Data['steering']
Odo_T_ms = Odo_Data['time']
print('Extracted!!!')