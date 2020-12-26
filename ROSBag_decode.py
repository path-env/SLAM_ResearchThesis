# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:20:24 2020

@author: MangalDeep
"""

import rosbag
from FastSLAM2_Proposal_W_Obsv import FastSLAM2
from squaternion import Quaternion
import matplotlib.pyplot as plt
from ParticleFilter import RBPF_SLAM
import numpy as np


bag = rosbag.Bag('G:/DataSets/CARLA_Autopilot_ROS.bag')

#FS = FastSLAM2()
RBPF = RBPF_SLAM()

#Topics in bag
Topics =  [*bag.get_type_and_topic_info()[1]]

'''
'/carla/actor_list', '/carla/ego_vehicle/gnss/gnss1/fix', 
'/carla/ego_vehicle/imu/imu1', '/carla/ego_vehicle/lidar/lidar1/point_cloud',
 '/carla/ego_vehicle/objects', '/carla/ego_vehicle/odometry',
 '/carla/ego_vehicle/semantic_lidar/lidar1/point_cloud', 
 '/carla/ego_vehicle/vehicle_info', '/carla/ego_vehicle/vehicle_status', 
 '/carla/marker', '/carla/objects', '/carla/status', '/carla/traffic_lights', 
 '/carla/traffic_lights_info', '/carla/world_info', '/clock', '/rosout', '/tf'
'''
## Initialization
Meas_X_t = {}
Meas_Z_t = {}
Meas_X_t_1 = {}
GPS_Z_t = {}
IMU_Z_t = {}
Lat = []
Long = []

Gps_avail ,Imu_avail,Lidar_avail ,Odom_avail,old_t = 0,0 ,0,0,0

for topic, msg, t in bag.read_messages(topics=['/carla/ego_vehicle/gnss/gnss1/fix',
                                               '/carla/ego_vehicle/odometry',
                                               '/carla/ego_vehicle/lidar/lidar1/point_cloud',
                                               '/carla/ego_vehicle/imu/imu1']):
    
    if old_t-t.to_sec() != 0:
        Gps_avail ,Imu_avail,Lidar_avail ,Odom_avail = 0 ,0,0,0 
        
    if topic == '/carla/ego_vehicle/odometry':  
        Quat = Quaternion(msg.pose.pose.orientation.w,msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z)
        Eul = Quat.to_euler(degrees = True)  #(roll, pitch, yaw)
        Meas_X_t['x'] = msg.pose.pose.position.x
        Meas_X_t['y'] = msg.pose.pose.position.y
        Meas_X_t['yaw'] = Eul[2]
        Meas_X_t['t']= t.to_sec()
        Odom_avail =1
        print(f"Odometry for {t.to_sec()} extracted")
    
    if topic == '/carla/ego_vehicle/lidar/lidar1/point_cloud':
        '''
        From sensor.json file
         "spawn_point": {"x": 0.0, "y": 0.0, "z": 2.4, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
         "range": 50,
        '''
        temp = np.frombuffer(msg.data,np.float32).reshape(-1,4)
        Meas_Z_t = {'x': temp[:,0],'y': -1*temp[:,1],'z': temp[:,2] } ## In sensor Frame
        Lidar_avail =1
        print(f"Lidar pointcloud for {t.to_sec()} extracted")
    
    if topic == '/carla/ego_vehicle/gnss/gnss1/fix':
        GPS_Z_t['lat']= (msg.latitude)
        GPS_Z_t['long']= (msg.longitude)
        GPS_Z_t['alt'] = msg.altitude
        GPS_Z_t['t']= t.to_sec()
        Gps_avail =1
        print(f"GNSS for {t.to_sec()} extracted") 
    
    if topic == '/carla/ego_vehicle/imu/imu1':
        Quat = Quaternion(msg.orientation.w ,msg.orientation.x,msg.orientation.y,msg.orientation.z)
        Eul = Quat.to_euler(degrees = True)  #(roll, pitch, yaw)
        IMU_Z_t['roll'] =  Eul[0]
        IMU_Z_t['pitch'] = Eul[1]
        IMU_Z_t['yaw'] =  Eul[2]
        IMU_Z_t['t']= t.to_sec()
        orientation = {'x':msg.orientation.x , 'y':msg.orientation.y , 'z':msg.orientation.z, 'w':msg.orientation.w }
        # Ang_vel = {'x':msg.angular_velocity.x , 'y':msg.angular_velocity.y , 'z':msg.angular_velocity.z }
        # lin_acc =  {'x':msg.linear_acceleration.x , 'y':msg.linear_acceleration.y , 'z':msg.linear_acceleration.z }
        Imu_avail =1
        print(f"IMU for {t.to_sec()} extracted") 
     
    ## Sync Time
    # if (Imu_avail and Gps_avail and Lidar_avail and Odom_avail)!=1:
    #     Gps_avail ,Imu_avail,Lidar_avail ,Odom_avail = 0 ,0,0,0
        
    if Imu_avail and Gps_avail and Odom_avail and Lidar_avail:            
        #FS.run(Meas_X_t_1,Meas_X_t,Meas_Z_t)    
        RBPF.Run(Meas_X_t,Meas_Z_t, GPS_Z_t,IMU_Z_t)
        print(f"Time { t.to_sec()} processed")
        Gps_avail ,Imu_avail,Lidar_avail ,Odom_avail = 0 ,0,0,0
    
    old_t = t.to_sec()
    
bag.close()


'''
Gps_avail ,Imu_avail,Lidar_avail ,Odom_avail,old_t = 0,0 ,0,0,0

for topic, msg, t in bag.read_messages(topics=['/carla/ego_vehicle/gnss/gnss1/fix','/carla/ego_vehicle/lidar/lidar1/point_cloud',
                                               '/carla/ego_vehicle/imu/imu1','/carla/ego_vehicle/odometry']):
    if old_t-t.to_sec() != 0:
        Gps_avail ,Imu_avail = 0 ,0          
    
    if topic == '/carla/ego_vehicle/gnss/gnss1/fix':
        GPS_Z_t['lat']= (msg.latitude)
        GPS_Z_t['long']= (msg.longitude)
        GPS_Z_t['alt'] = msg.altitude
        Meas_X_t['x'] =  GPS_Z_t['long']
        Meas_X_t['y'] =  GPS_Z_t['lat']
        Meas_X_t['t'] = t.to_sec()
        Gps_avail = 1
        print(f"GNSS for intialization {t.to_sec()} extracted")
        
    if topic == '/carla/ego_vehicle/imu/imu1':
        Quat = Quaternion(msg.orientation.w ,msg.orientation.x,msg.orientation.y,msg.orientation.z)
        Eul = Quat.to_euler(degrees = True)  #(roll, pitch, yaw)
        IMU_Z_t['roll'] =  Eul[0]
        IMU_Z_t['pitch'] = Eul[1]
        IMU_Z_t['yaw'] =  Eul[2]
        Meas_X_t['yaw'] =  IMU_Z_t['yaw']        
        orientation = {'x':msg.orientation.x , 'y':msg.orientation.y , 'z':msg.orientation.z, 'w':msg.orientation.w }
        # Ang_vel = {'x':msg.angular_velocity.x , 'y':msg.angular_velocity.y , 'z':msg.angular_velocity.z }
        # lin_acc =  {'x':msg.linear_acceleration.x , 'y':msg.linear_acceleration.y , 'z':msg.linear_acceleration.z }
        Imu_avail = 1
        print(f"IMU for {t.to_sec()} extracted") 
    
    if topic == '/carla/ego_vehicle/lidar/lidar1/point_cloud':
        temp = np.frombuffer(msg.data,np.float32).reshape(-1,4)
        Meas_Z_t = {'X': temp[:,0],'Y': temp[:,1],'Z': temp[:,2] }
        Lidar_avail =1
        print(f"Lidar pointcloud for {t.to_sec()} extracted")
    # if topic == '/carla/ego_vehicle/odometry':  
    #     Quat = Quaternion(msg.pose.pose.orientation.w,msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z)
    #     Eul = Quat.to_euler(degrees = True)  #(roll, pitch, yaw)
    #     Meas_X_t['x'] = msg.pose.pose.position.x
    #     Meas_X_t['y'] = msg.pose.pose.position.y
    #     Meas_X_t['yaw'] = Eul[2]
    #     print(f"Odometry for {t.to_sec()} extracted")
     
    old_t = t.to_sec()
    # if (Gps_avail and Imu_avail)!=1:
    #     Gps_avail ,Imu_avail = 0 ,0
    if Gps_avail and Imu_avail and Lidar_avail:
        RBPF.Initialize(GPS_Z_t,IMU_Z_t,Meas_Z_t)
        Init_time= t.to_sec()
        Gps_avail ,Imu_avail = 0 ,0
        break
'''
