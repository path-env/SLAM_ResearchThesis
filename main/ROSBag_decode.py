# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:20:24 2020

@author: MangalDeep
"""

import logging
# To resolve VS code error
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join('..', 'SLAM_ResearchThesis')))

import numpy as np
import rosbag
from libs.slam_analyzer import Analyze as aly
# from slam_posegraph.graph_constructor import Graph
# from slam_posegraph.graph_optimizer import ManifoldOptimizer
from slam_particlefilter.particle_filter import RBPF_SLAM
from slam_particlefilter.gmapping import Gmapping
from squaternion import Quaternion

def ROS_bag_run():
    if sys.platform =='linux':
        bag = rosbag.Bag('/media/mangaldeep/HDD3/DataSets/Bagfiles/CARLA_Autopilot_ROS_01_02_2021_Town3.bag') #508 - 620
        #bag = rosbag.Bag('/media/mangaldeep/HDD3/DataSets/Bagfiles/CARLA_Autopilot_ROS_08_02_2021_mountain.bag')
    else:
        bag = rosbag.Bag('G:/DataSets/BagFiles/CARLA_Autopilot_ROS.bag') #508 - 620

    plotter = aly('GMapping')
    slam_obj = Gmapping(plotter)
    #slam_obj = RBPF_SLAM(plotter)
    logger = logging.getLogger('ROS_Decode')

    #slam_obj = Graph()
    #slam_opt_obj = ManifoldOptimizer()

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
    GPS_Z_t = {}
    IMU_Z_t = {}
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
            Meas_X_t['acc'] = -1*np.sqrt(msg.acceleration.linear.x**2 +  msg.acceleration.linear.y**2 +msg.acceleration.linear.z**2)
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
            plotter.set_groundtruth(GPS_Z_t, IMU_Z_t, Meas_X_t)      
            #if (Meas_X_t['v']>0.01 or Meas_X_t['v'] <-0.01):
            ##  Graph Based    
            #slam_obj.create_graph(Meas_X_t , Meas_Z_t )

            ## Particle Based
            slam_obj.run(Meas_X_t,Meas_Z_t, GPS_Z_t,IMU_Z_t)
            #slam_obj.set_groundtruth( GPS_Z_t, IMU_Z_t, Meas_X_t)

            #slam_obj.run(Meas_X_t_1,Meas_X_t,Meas_Z_t)

            #logger.info(f"Time { t.to_sec()} processed")
            Gps_avail ,Imu_avail,Lidar_avail ,Odom_avail,Vel_avail,veh_info = 0, 0,0,0,0,1
        
        old_t = t.to_sec()
    slam_obj.plot_results()    
    #slam_opt_obj.optimize(Gp)
    bag.close()
    
if __name__ == '__main__':
    ROS_bag_run()
