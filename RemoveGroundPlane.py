"""
Created on Mon Jul  6 21:38:31 2020

@author: Mangal
"""

def Zbased(Meas_Z_t, LowerThreshold, UpperThreshold):
    import numpy as np
    Zindx = np.where(np.logical_and(Meas_Z_t['z']>=LowerThreshold ,Meas_Z_t['z']<=UpperThreshold))
    return Zindx
#%%
def RANSAC(Meas_Z_t, LowerThreshold, UpperThreshold):
    import random
    import math
    import numpy as np
    #RANSAC
    #sample
    Sample_S = 3
    # probability of success
    Success_prob =0.9
    
    outlier_ratio = (0.5*len(Meas_Z_t)/len(Meas_Z_t))
    
    #Number of trials
    Trials = math.log(1-Success_prob)/(math.log(1-(1-outlier_ratio)**Sample_S))
    PerpendicularVector = []
    InlierCount = []
    OutlierCount = []
    Outlier =[]
    for TrialCount in range(math.ceil(Trials)):
        RandomPoints = random.sample(range(len(Meas_Z_t)),3)
        #RandomPoints = [0,1,2]
        GP_inlier = [] 
        GP_outlier = []
        GP_OnthePlane = []

        #compute/Model building
        #Fit points to the plane 
        P = Meas_Z_t[RandomPoints[0]][:3]
        Q = Meas_Z_t[RandomPoints[1]][:3]
        R = Meas_Z_t[RandomPoints[2]][:3]
        # P= np.array([1,0,2]);
        # Q = np.array([-1,1,2]);
        # R = np.array([5,0,3]);
        PQ, PR = P-Q, P-R
        #c = np.array([5,0,3]);
        PerpendicularVector.append(np.cross(PQ ,PR))
        COeff =np.dot(PerpendicularVector[TrialCount],(P.T))
        OnThePlane_Threshold = np.dot(PerpendicularVector[TrialCount],Q) - COeff  
        den = (np.sqrt(np.sum(np.square(PerpendicularVector[TrialCount][0]) + np.square(PerpendicularVector[TrialCount][1]) + np.square(PerpendicularVector[TrialCount][2]))))
        for i in range(len(Meas_Z_t)):
            if np.array_equal(Meas_Z_t[i][3],Q):
                pass
            #print(f'Trial = {TrialCount}.......PointCloudIndx ={i}')
            #pointCheck = P - PointClouds_Reshaped[i][:3]
            #PlaneEq = PerpendicularVector[TrialCount][0].(x -P[0]) + PerpendicularVector[TrialCount][1].(y -P[1]) + PerpendicularVector[TrialCount][2].(z -P[2])

            num = abs(np.dot(PerpendicularVector[TrialCount],Meas_Z_t[i][:3]) - COeff)
            Dist2Plane = num/max(den,0.00000000001)
            if abs(Dist2Plane) <= OnThePlane_Threshold:
                GP_OnthePlane.append(i)
            elif abs(Dist2Plane) <=LowerThreshold:
                GP_inlier.append(i)
            else:
                GP_outlier.append(i)
        Outlier.append(GP_outlier)        
        InlierCount.append(len(GP_inlier))
       #InlierCount.append(len(GP_OnthePlane))
        OutlierCount.append(len(GP_outlier))
    
    #Find Plane with Max Inlier
    InlierCount = np.array(InlierCount)
    MaxInd =  np.where(InlierCount == max(InlierCount))[0].tolist()[0]
    
        #remove Points above Vehicles height
    Indx = np.where(Meas_Z_t[:,2]<UpperThreshold) 
    return    Indx.intersection(Outlier[MaxInd]) 
    #return Outlier[MaxInd]    
#%%
if __name__ == '__main__': 
    import numpy as np
    from nuscenes.nuscenes import NuScenes
    import platform
    import matplotlib.pyplot as plt
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import RadarPointCloud,LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points, transform_matrix
    #%%
    if platform.system() =='Linux':
        Dataloc = '/media/mangaldeep/HDD3/data/sets/nuscenes/v1.0-mini'
        nusc = NuScenes(version='v1.0-mini', dataroot=Dataloc, verbose=True)
    elif platform.system() == 'Windows':
        Dataloc = 'G:/data/sets/nuscenes/v1.0-mini'
        nusc = NuScenes(version='v1.0-mini', dataroot=Dataloc, verbose=True)
    #%%
    Scenes = nusc.list_scenes()
    #Selected_Scene = int(input('Enter the scene to remove Ground reflections:'))
    Selected_Scene =1;
    scene = nusc.scene[Selected_Scene]
    NoOfSamples = scene['nbr_samples']    
    scenetoken = scene['first_sample_token']
    sample = nusc.get('sample',scenetoken)
    SampleCount = 1
    #occupancy grid params
    X_pos = []
    Y_pos = []
    while sample['next']:
        LidarData = nusc.get('sample_data',sample['data']['LIDAR_TOP'])
        Ego_pose = nusc.get('ego_pose',LidarData['ego_pose_token'])
        Ego_Position = Ego_pose['translation'][0:2]
        Ego_rotation = Quaternion(Ego_pose['rotation']).yaw_pitch_roll
        Ego_yaw = -np.degrees(Ego_rotation[0])
        X_pos.append(Ego_Position[0])
        Y_pos.append(Ego_Position[1])
        sample =  nusc.get('sample',sample['next'])
    
    Map_init = Map(X_pos,Y_pos,1,scene['name'])    
    scenetoken = scene['first_sample_token'] 
    sample = nusc.get('sample',scenetoken)  
    
    while SampleCount <= scene['nbr_samples']:
         # LIDAR processing
         LidarData = nusc.get('sample_data',sample['data']['LIDAR_TOP'])
         Ego_pose = nusc.get('ego_pose',LidarData['ego_pose_token'])
         Lidar_LogFilepath = LidarData['filename']
         LIDAR_PointClouds_raw = LidarPointCloud.from_file((Dataloc+'/'+Lidar_LogFilepath))
         nusc.render_sample_data(LidarData['token'])
         #nusc.render_pointcloud_in_image(sample['token'], pointsensor_channel='LIDAR_TOP')
        
         #transform to vehicle frame
         Lidar_calibration=nusc.get('calibrated_sensor',LidarData['calibrated_sensor_token']) 
         Lidar_Trans_Matrix = transform_matrix(Lidar_calibration['translation'],Quaternion(Lidar_calibration['rotation']), inverse=False)
         LIDAR_PointClouds_raw.transform(Lidar_Trans_Matrix)
         Lidar_PC = LIDAR_PointClouds_raw.points.T
         Meas_Z_t =  Lidar_PC[:,[0,1,2,3]]
        
         #EgoPoses
         Ego_Position = Ego_pose['translation'][0:2]
         Ego_rotation = Quaternion(Ego_pose['rotation']).yaw_pitch_roll
         Ego_yaw = -np.degrees(Ego_rotation[0])
         Pose_X_t = [Ego_Position[0],Ego_Position[1], Ego_yaw]
        
         #RANSAC
         #GroundPlaneRemovedIndices = RANSAC(Meas_Z_t,2)
         GroundPlaneRemovedIndices = Zbased(Meas_Z_t,1)
    
         #Measurement preprocessing
         Meas_Z_t = Meas_Z_t[GroundPlaneRemovedIndices]
         Azimuth = np.degrees(np.arctan2(Meas_Z_t[:,1], Meas_Z_t[:,0]).reshape((len(Meas_Z_t),-1)))
         Meas_Z_t = np.hstack([Meas_Z_t,Azimuth]) #Measurement Azimuth in degrees
        # Ring15 = np.where((Meas_Z_t[3,:]==10))
        # Meas_Z_t = Meas_Z_t.reshape((5,-1)) #[:,Ring15]
        
         #remove ground reflections and moving objects -O/P of task 3
         Map = Map_init.Update(Pose_X_t,Meas_Z_t,SampleCount)  #LocalMap
         #GridMap(Grid_Map,Pose_X_t,Meas_Z_t.T)
        
        #Plotting
         plt.figure()
         plt.clf()
         #plt.ion()
         
         plt.grid(True,which='both')
         circle = plt.Circle((Pose_X_t[0] ,Pose_X_t[1] ), radius=3.0, fc='y')
         #arrow = np.array([Pose_X_t[0] ,Pose_X_t[1]]) + np.array([3.5, 0]).dot(np.array([[np.cos(Pose_X_t[2]), np.sin(Pose_X_t[2])], [-np.sin(Pose_X_t[2]), np.cos(Pose_X_t[2])]]))
         plt.gca().add_patch(circle)        
         #plt.plot([Pose_X_t[1] , arrow[1]], [Pose_X_t[0] , arrow[0]])
         # img = ndimage.rotate(1.0 - 1./(1.+np.exp(Map)),180)
         plt.imshow(1.0 - 1./(1.+np.exp(Map)), cmap='Greys')
         plt.draw()
         plt.show() 
         plt.imsave(scene['name']+'_sample'+str(SampleCount)+('.png'),Map, format='png',cmap = 'Greys')
              
          
         print(f'Sample {SampleCount} processed!!')
         SampleCount+=1
         if sample['next']:          
            sample =  nusc.get('sample',sample['next'])
    assert SampleCount-1 ==NoOfSamples, 'Not all samples read'
#%% sort by ring index
# # points = scan.reshape((-1, 5))
# indx =np.where(LIDAR_PointClouds_Reshaped[:,4]==31)
# ring2 = LIDAR_PointClouds_Reshaped[indx]
# rotate_mat = np.array([[1,0],[0,-1]]) 
# ring2 =np.matmul(ring2[:,0:2],rotate_mat)
# #plt.scatter(ring2[:,0],ring2[:,1])



    
    
