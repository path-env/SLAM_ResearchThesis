# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:20:24 2020

@author: MangalDeep
"""

# To resolve VS code error
import sys
from pathlib import Path
sys.path[0] = str(Path('G:\Masters-FHD\Sem3\SLAM_ResearchThesis'))

import rosbag
from squaternion import Quaternion
#import matplotlib.pyplot as plt
import numpy as np
import logging

# from slam_posegraph.graph_constructor import Graph
# from slam_posegraph.graph_optimizer import ManifoldOptimizer
# from slam_particlefilter.gmapping import FastSLAM2
from slam_particlefilter.particle_filter import RBPF_SLAM

def ROS_bag_run():
    bag = rosbag.Bag('G:/DataSets/BagFiles/CARLA_Autopilot_ROS.bag') #508 - 620
    
    #FS = FastSLAM2()
    # #Logging
    logger = logging.getLogger('ROS_Decode')
    # logger.setLevel(logging.DEBUG)
    
    # fh = logging.FileHandler('ROS_Decode.log')
    # fh.setLevel(logging.DEBUG)
    
    # # create console handler with a higher log level
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.WARNING)
    
    # # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    
    # # add the handlers to the logger
    # logger.addHandler(fh)
    #logger.addHandler(ch)
    RBPF = RBPF_SLAM()
    #Gp = Graph()
    #Opt = ManifoldOptimizer()
    #Topics in bag
    #Topics =  [*bag.get_type_and_topic_info()[1]]
    
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
    g= []
    Gps_avail ,Imu_avail,Lidar_avail ,Odom_avail,old_t = 0,0 ,0,0,0
    max_steering_angle = 1.221730351448059
    for topic, msg, t in bag.read_messages(topics=['/carla/ego_vehicle/gnss/gnss1/fix',
                                                   '/carla/ego_vehicle/odometry',
                                                   '/carla/ego_vehicle/lidar/lidar1/point_cloud',
                                                   '/carla/ego_vehicle/vehicle_info',
                                                   '/carla/ego_vehicle/vehicle_status',
                                                   '/carla/ego_vehicle/imu/imu1']):
        
        if old_t-t.to_sec() != 0:
            Gps_avail ,Imu_avail,Lidar_avail ,Odom_avail,Vel_avail,veh_info = 0, 0,0,0,0,1
            
        if  topic == '/carla/ego_vehicle/vehicle_status': 
            Meas_X_t['v'] = msg.velocity
            Meas_X_t['acc'] = msg.acceleration.linear.x
            Meas_X_t['steer'] = msg.control.steer * max_steering_angle # on a scale [-1,1] of max_steering_angle
            Vel_avail =1
            logger.info(f"Velocity for {t.to_sec()} extracted")
            
        # if  topic == '/carla/ego_vehicle/vehicle_info':
        #     veh_info = 1
        #     print('/carla/ego_vehicle/vehicle_info')
                
        if topic == '/carla/ego_vehicle/odometry':  
            Quat = Quaternion(msg.pose.pose.orientation.w,msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z)
            Eul = Quat.to_euler(degrees = True)  #(roll, pitch, yaw)
            Meas_X_t['x'] = msg.pose.pose.position.x + np.random.randn()*1
            Meas_X_t['y'] = msg.pose.pose.position.y + np.random.randn()*1
            Meas_X_t['yaw'] = Eul[2] + np.random.randn()*1
            Meas_X_t['t']= t.to_sec()
            #g.append(t.to_sec())
            Odom_avail =1
            logger.info(f"Odometry for {t.to_sec()} extracted")
        
        if topic == '/carla/ego_vehicle/lidar/lidar1/point_cloud':
            '''
            From sensor.json file
              "spawn_point": {"x": 0.0, "y": 0.0, "z": 2.4, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
              "range": 50,
            '''
            temp = np.frombuffer(msg.data,np.float32).reshape(-1,4)
            Meas_Z_t = {'x': temp[:,0],'y': -1*temp[:,1],'z': temp[:,2] } ## In sensor Frame
            Lidar_avail =1
            logger.info(f"Lidar pointcloud for {t.to_sec()} extracted")
        
        if topic == '/carla/ego_vehicle/gnss/gnss1/fix':
            GPS_Z_t['lat']= (msg.latitude)
            GPS_Z_t['long']= (msg.longitude)
            GPS_Z_t['alt'] = msg.altitude
            GPS_Z_t['t']= t.to_sec()
            Gps_avail =1
            logger.info(f"GNSS for {t.to_sec()} extracted") 
        
        if topic == '/carla/ego_vehicle/imu/imu1':
            Quat = Quaternion(msg.orientation.w ,msg.orientation.x,msg.orientation.y,msg.orientation.z)
            Eul = Quat.to_euler(degrees = True)  #(roll, pitch, yaw)
            IMU_Z_t['roll'] =  Eul[0]
            IMU_Z_t['pitch'] = Eul[1]
            IMU_Z_t['yaw'] =  Eul[2]
            IMU_Z_t['t']= t.to_sec()
            #orientation = {'x':msg.orientation.x , 'y':msg.orientation.y , 'z':msg.orientation.z, 'w':msg.orientation.w }
            Ang_vel = {'x':msg.angular_velocity.x , 'y':msg.angular_velocity.y , 'z':msg.angular_velocity.z }
            IMU_Z_t['ang_vel'] = Ang_vel['z']
            # lin_acc =  {'x':msg.linear_acceleration.x , 'y':msg.linear_acceleration.y , 'z':msg.linear_acceleration.z }
            Imu_avail =1
            logger.info(f"IMU for {t.to_sec()} extracted") 
            
        if Imu_avail and Gps_avail and Odom_avail and Lidar_avail and Vel_avail and veh_info:            
            #FS.run(Meas_X_t_1,Meas_X_t,Meas_Z_t)
            #if (Meas_X_t['v']>0.01 or Meas_X_t['v'] <-0.01):
                
            #Gp.create_graph(Meas_X_t , Meas_Z_t )
            #RBPF.run(Meas_X_t,Meas_Z_t, GPS_Z_t,IMU_Z_t)
            #logger.info(f"Time { t.to_sec()} processed")
            RBPF.set_groundtruth( GPS_Z_t, IMU_Z_t, Meas_X_t)
    
            Gps_avail ,Imu_avail,Lidar_avail ,Odom_avail,Vel_avail,veh_info = 0, 0,0,0,0,1
        
        old_t = t.to_sec()
    RBPF.plot_results()    
    #Gp.plot()
    #Opt.optimize(Gp)
    bag.close()
    
if __name__ == '__main__':
    ROS_bag_run()