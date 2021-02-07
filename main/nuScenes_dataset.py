# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:29:43 2020

@author: MangalDeep
"""
# To resolve VS code error
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
from nuscenes.nuscenes import NuScenes
import platform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from RemoveGroundPlane import RANSAC,Zbased
from OccupancyGrid import Map
#%%
if platform.system() =='Linux':
    Dataloc = '/media/mangaldeep/HDD3/data/sets/nuscenes/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=Dataloc, verbose=True)
elif platform.system() == 'Windows':
    nusc_can = NuScenesCanBus(dataroot='G:/DataSets/Nuscenes/data/sets/nuscenes')
    Dataloc = 'G:/DataSets/Nuscenes/data/sets/nuscenes/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=Dataloc, verbose=True)
#%%
Scenes = nusc.list_scenes()
Selected_Scene = int(input('Enter the scene to remove Ground reflections:'))
#Selected_Scene =0
scene = nusc.scene[Selected_Scene]
# nusc_can.print_all_message_stats(scene['name'])
# nusc_can.get_messages(scene['name'], 'zoe_veh_info')
NoOfSamples = scene['nbr_samples']    
scenetoken = scene['first_sample_token']
sample = nusc.get('sample',scenetoken)
SampleCount = 1
X_pos = []
Y_pos = []
Ego_yaw = []
while sample['next']:
    LidarData = nusc.get('sample_data',sample['data']['LIDAR_TOP'])
    Ego_pose = nusc.get('ego_pose',LidarData['ego_pose_token'])
    Ego_Position = Ego_pose['translation'][0:2]
    Ego_rotation = Quaternion(Ego_pose['rotation']).yaw_pitch_roll
    Ego_yaw.append(-np.degrees(Ego_rotation[0]))
    X_pos.append(Ego_Position[0])
    Y_pos.append(Ego_Position[1])
    sample =  nusc.get('sample',sample['next'])
    Poses = {'X': X_pos,'Y': Y_pos,'Orientation':Ego_yaw}

Map_init = Map(Poses,1,scene['name'])    
scenetoken = scene['first_sample_token'] 
sample = nusc.get('sample',scenetoken)  

while SampleCount <= scene['nbr_samples']:
     # LIDAR processing
     LidarData = nusc.get('sample_data',sample['data']['LIDAR_TOP'])
     Ego_pose = nusc.get('ego_pose',LidarData['ego_pose_token'])
     Lidar_LogFilepath = LidarData['filename']
     LIDAR_PointClouds_raw = LidarPointCloud.from_file((Dataloc+'/'+Lidar_LogFilepath))
     #nusc.render_sample_data(LidarData['token'])
     #nusc.render_pointcloud_in_image(sample['token'], pointsensor_channel='LIDAR_TOP')
    
     #transform to vehicle frame
     Lidar_calibration=nusc.get('calibrated_sensor',LidarData['calibrated_sensor_token']) 
     Lidar2Ego_Trans_Matrix = transform_matrix(Lidar_calibration['translation'],Quaternion(Lidar_calibration['rotation']), inverse=False)
     LIDAR_PointClouds_raw.transform(Lidar2Ego_Trans_Matrix)
     # LIDAR_PointClouds_raw.rotate(Quaternion(Lidar_calibration['rotation']).rotation_matrix)
     # LIDAR_PointClouds_raw.translate(np.array(Lidar_calibration['translation']))
     
     #tranform to global frame
     # Ego2Global_Trans_Matrix = transform_matrix(Ego_pose['translation'],Quaternion(Ego_pose['rotation']), inverse=False)
     # LIDAR_PointClouds_raw.transform(Ego2Global_Trans_Matrix)
     # LIDAR_PointClouds_raw.rotate(Quaternion(Ego_pose['rotation']).rotation_matrix)
     # LIDAR_PointClouds_raw.translate(np.array(Ego_pose['translation']))   
     Lidar_PC = LIDAR_PointClouds_raw.points.T
     LIDAR_PointClouds_inEGO =  Lidar_PC[:,[0,1,2,3]]
    
     #EgoPoses
     Ego_Position = Ego_pose['translation'][0:2]
     Ego_rotation = Quaternion(Ego_pose['rotation']).yaw_pitch_roll
     Ego_yaw = -np.degrees(Ego_rotation[0])
     Pose_X_t = {'x':Ego_Position[0] ,'y':Ego_Position[1] ,'yaw':Ego_yaw  }
    
     #RANSAC
     #GroundPlaneRemovedIndices = RANSAC(LIDAR_PointClouds_inEGO,2)

     #Measurement preprocessing
     #remove ground reflections and moving objects -O/P of task 3
     Meas_Z_t = LIDAR_PointClouds_inEGO
    # Ring15 = np.where((Meas_Z_t[3,:]==10))
    # Meas_Z_t = Meas_Z_t.reshape((5,-1)) #[:,Ring15]
    
     Meas_Z_t = {'x': Meas_Z_t[:,0],'y': Meas_Z_t[:,1],'z': Meas_Z_t[:,2] }
     Map = Map_init.Update(Pose_X_t,Meas_Z_t, PltEnable = True)  #LocalMap
     #GridMap(Grid_Map,Pose_X_t,Meas_Z_t.T)
    
    #Plotting
     fig2, (ax) = plt.subplots()
     #plt.ion()
      # img = (nusc.render_sample_data(LidarData['token']))
      # img =  mpimg.imread(img)
      # ax1.imshow(img)
     plt.grid(True,which='both')
     # plt.xlim(Map_init.Xlim_start,Map_init.Xlim_end)
     # plt.ylim(Map_init.Ylim_start,Map_init.Ylim_end)
     Veh = patches.Rectangle((Pose_X_t['x']-Map_init.Xlim_start -3.5 , Pose_X_t['y']-Map_init.Ylim_start-5),10,7,linewidth=1,edgecolor='r')
      # t_start = ax.transData
      # t = mpl.transforms.Affine2D().rotate_deg(Pose_X_t[2])
      # t_end = t_start + t
      #Veh.set_transform(t_end)
    #arrow = np.array([Pose_X_t[0] ,Pose_X_t[1]]) + np.array([3.5, 0]).dot(np.array([[np.cos(Pose_X_t[2]), np.sin(Pose_X_t[2])], [-np.sin(Pose_X_t[2]), np.cos(Pose_X_t[2])]]))
     ax.add_patch(Veh)        
      #plt.plot([Pose_X_t[1] , arrow[1]], [Pose_X_t[0] , arrow[0]])  
     #plt.imshow(1.0 - 1./(1.+np.exp(Map)), 'Greys')
     probMap = np.exp(Map)/(1.+np.exp(Map))                    
     ax2 = plt.imshow(probMap, cmap='Greys')
     plt.draw()
     plt.show() 
      #plt.imsave(scene['name']+'_sample'+str(SampleCount)+('.png'),Map, format='png',cmap = 'Greys')
               
     print(f'Sample {SampleCount} processed!!')
     SampleCount+=1
     if sample['next']:          
        sample =  nusc.get('sample',sample['next'])
assert SampleCount-1 ==NoOfSamples, 'Not all samples read'