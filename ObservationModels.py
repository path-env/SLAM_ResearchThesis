# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:42:48 2020

@author: MangalDeep
"""
import numpy as np
import pandas as pd
from RemoveGroundPlane import RANSAC,Zbased
##################################### Measurement Models ################################################
    
def Lidar_Observation_Model(Est_X_t , Meas_Z_t ):
    #Linearize
    Est_x = sp.Symbol('Est_x')
    Est_y = sp.Symbol('Est_y')
    Est_ori = sp.Symbol('Est_ori')
    Meas_x = sp.Symbol('Meas_x')
    Meas_y = sp.Symbol('Meas_y')
    
    WRT = Matrix([Est_x , Est_y, Est_ori ])
    Est_Z_t = Matrix([((Meas_x - Est_x)**2 + (Meas_y - Est_y)**2)**0.5 , ( atan2(Est_y - Meas_y , Est_x - Meas_x ) - Est_ori )] )
    Meas_Jacb_Z_t = Est_Z_t.jacobian(WRT)  
    
    Meas_Jacb_Z_t.subs( { 
                        Est_x: Est_X_t['x'],
                        Est_y: Est_X_t['y'],
                        Est_ori: Est_X_t['yaw'],
                        Meas_x: Meas_Z_t['x'],
                        Meas_y: Meas_Z_t['y']   })
    
    #Without Linearization
    #Range = np.hypot((Est_x - Meas_x) , (Est_y - Meas_y))
    #Azimuth = np.rad2deg( np.arctan2( (Est_y - Meas_y) , (Est_x - Meas_x) ) )
    
def CoordinateTransform_t_1(Meas_Z_t,Meas_X_t,Meas_X_t_1):
    trans_Z_t = {}
    X = Meas_Z_t['x']
    Y = Meas_Z_t['y']
    Z = np.array([X,Y]).reshape(2,-1)
    dx = Meas_X_t['x'] - Meas_X_t_1['x']
    dy = Meas_X_t['y'] - Meas_X_t_1['y']
    T = np.array([dx,dy]).reshape(2,1)
    dyaw =np.deg2rad(Meas_X_t['yaw'] - Meas_X_t_1['yaw'])
    R = np.array([[np.cos(dyaw),-np.sin(dyaw)] , [np.sin(dyaw),np.cos(dyaw)]]).reshape(2,2)
    Meas_t = R@Z + T
    trans_Z_t['x'] = Meas_t[0,:]
    trans_Z_t['y'] = Meas_t[1,:]
    trans_Z_t['z'] = Meas_Z_t['z'].to_numpy()
    return trans_Z_t    

def Likelihood_Field_Observation_Model(Meas_Z_t , Est_X_t , Map_X_t_1,Meas_X_t_1,Map_obj,LidarPose_X_t = {'x':0,'y':0 , 'yaw':0} ,z_hit = 0.5 ,sigma= 1, z_random=0.3 , z_max= 0.2):
    likly = 1
   
    '''    
    #Sensor_Pose_inHost = np.array([[LidarPose_X_t['x']] , [ LidarPose_X_t['y']]]).T
    # from sensors.json file
    #"spawn_point": {"x": 0.0, "y": 0.0, "z": 2.4, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    # Meas_Z_t_nparr = Meas_Z_t.to_numpy()
    # Est_X_t_nparr = np.array(list(Est_X_t.values()))
    # Range_Z =  Meas_Z_t_nparr[:,4]
    # Azimuth_Z = Meas_Z_t_nparr[:,3]
    # Meas_loc_Vector = np.array([ np.cos(np.deg2rad(Azimuth_Z + LidarPose_X_t['yaw'])) , np.sin(np.deg2rad(Azimuth_Z + LidarPose_X_t['yaw'] ))])
    # # Signs swapped in the Rotation matrix in a12 and a21 due to numpy issues -- Need to fix it
    # Rot_Matrix = np.array( [ [np.cos(np.deg2rad(Est_X_t_nparr[2])) , np.sin(np.deg2rad(Est_X_t_nparr[2]))]  , [ -np.sin(np.deg2rad(Est_X_t_nparr[2])) , np.cos(np.deg2rad(Est_X_t_nparr[2])) ]]).T
    # MeasGlobal_Z_t_Vec = Est_X_t_nparr[:2].reshape(2,1) + (Rot_Matrix @ Sensor_Pose_inHost).reshape(2,-1) + Range_Z * Meas_loc_Vector
    # Closest_x , Closest_y = np.argmin(np.abs(np.subtract(MeasGlobal_Z_t_Vec,Map_X_t_1))) 
    # dist = ( (MeasGlobal_Z_t_Vec[0] - Closest_x )**2 ) + ((MeasGlobal_Z_t_Vec[1] - Closest_y )**2 )
'''
 
    trans_Z_t = CoordinateTransform_t_1(Meas_Z_t,Est_X_t,Meas_X_t_1)
    if 'Range_XY_plane' not in trans_Z_t.keys():
        trans_Z_t = Map_obj.Lidar_3D_Preprocessing(trans_Z_t)    
    Azi_rads = np.arctan2(trans_Z_t['y'],trans_Z_t['x'])
    Est_X_t['yaw'] = np.deg2rad(Est_X_t['yaw'])
    Sensor_Pose_inHost = np.array([[LidarPose_X_t['x']] , [ LidarPose_X_t['y']]])
    #Find Occupied cells
    # For Occupancy grid from __inverse_sensor_model >= 0.7 are occupied and <=0.3 free and 0.5= unknown
    # For Occupancy grid from __inverse_sensor_model2 +ve are occupied and -ve are free and 0 = unknown
    Map = Map_X_t_1.reshape(Map_obj.Lat_Width,Map_obj.Long_Length)
    Map_state = np.where((Map>0))
    Host_Pose = np.array([Est_X_t['x'] ,Est_X_t['y']]).reshape(2,1)
    CG =    Map_obj.getMapPivotPoint()
    for i in range(len(trans_Z_t['x'])):
        #Rot_Matrix = np.array( [np.cos(Est_X_t['yaw']) , -np.sin(Est_X_t['yaw']), np.sin(Est_X_t['yaw']) , np.cos(Est_X_t['yaw'])]).reshape(2,2)
        Meas_in_Host = -1*trans_Z_t['Range_XY_plane'][i]*np.array([np.cos(Azi_rads[i] + Est_X_t['yaw']),np.sin(Azi_rads[i] + Est_X_t['yaw']  )]).reshape(2,1)
        
        #Meas_Global = Host_Pose + (Rot_Matrix  @ Sensor_Pose_inHost) + Meas_in_Host
        Meas_Global = Host_Pose + Meas_in_Host

        #Laser reading to Map quadrants in the Matrix format
        Meas_in_Matrix_x = (CG[0] - Meas_Global[0])
        Meas_in_Matrix_y = (CG[1] - Meas_Global[1])
        MapIndx_x = Map_state[0][np.argmin(np.abs(Meas_in_Matrix_x - Map_state[0]))]
        MapIndx_y = Map_state[1][np.argmin(np.abs(Meas_in_Matrix_y - Map_state[1]))]
        
        dist = min(( (CG[0] - Meas_Global[0] - MapIndx_x)**2) + min((CG[1] - Meas_Global[1] - MapIndx_y)**2 ))
        temp = ( z_hit*Prob_Gaus_dist(dist,sigma) + (z_random/z_max))
        likly *=temp
    return likly

def Prob_Gaus_dist(MeanCentered , Var):
    return ((np.sqrt(2*np.pi*Var)**-1) *np.exp( (-0.5* MeanCentered**2)/Var))

######################################
def Lidar_3D_Preprocessing(Meas_Z_t):
    '''
    Parameters
    ----------
    Meas_Z_t : Dict with keys X, Y, Z , Ring Indices
        Input in cartisean coordinate system.
    Returns
    -------
    Meas_Z_t : Pandas
        DESCRIPTION:
        Find Azimuth -180 to 180 degrees and round it to 0.1
        Find Range in XY_plane
        Sort it with Azimuth and Range_XY_plane (descending) and remove the duplicates. Reset the indices last
    '''
    #RANSAC
    #GroundPlaneRemovedIndices = RANSAC(Meas_Z_t,2)
    #Req_Indices = Zbased(Meas_Z_t,-2.2,5)
    #Meas_Z_t = {'x':Meas_Z_t['x'][Req_Indices] ,'y':Meas_Z_t['y'][Req_Indices], 'z':Meas_Z_t['z'][Req_Indices]}
    Meas =  pd.DataFrame(Meas_Z_t)
    Meas = Meas.assign(Azimuth = np.rad2deg(np.arctan2(Meas['y'].values , Meas['x'].values)))
    Meas = Meas.assign(Range_XY_plane = np.hypot(Meas['x'].values,Meas['y'].values))
    Meas  = Meas.sort_values(by=['Azimuth','Range_XY_plane'],ascending=False)
    Meas = Meas.round({'Azimuth':1})
    Meas = Meas.drop_duplicates(['Azimuth'])
    Meas_Z_t = Meas.reset_index(drop=True)
    return Meas_Z_t

###################################### Incremental_ML_Mapping 14.1 ##############################################
def incrementalMLMapping(Meaz_Z_t , In_U_t , Updtd_X_t_1 , Map):
    s = Odometry_Motion_Model_Sample( Meas_X_t_1, Meas_X_t ,Updtd_X_t_1)
    
    s = s+ k*( Likelihood_Field_Observation_Model_derivative()  + Odometry_Motion_Model_Sample_derivative( ))
    
def Likelihood_Field_Observation_Model_derivative(Est_X_t, Meas_Z_t , Map_X_t_1):
    LidarPose_X_t = {'x':1 ,'y':1  }
    #Meas_Z_t = Lidar_3D_Preprocessing(Meas_Z_t)
    Meas_Z_t_nparr = Meas_Z_t
    Est_X_t_nparr = np.array(list(Est_X_t.values()))
    d_x,d_y,d_ori = 0,0,0
    Sensor_Pose_inHost = np.array([[LidarPose_X_t['x']] , [ LidarPose_X_t['y']]])
    Range_Z =  Meas_Z_t_nparr[:,3]
    Azimuth_Z = Meas_Z_t_nparr[:,4]
    Meas_loc_Vector = np.array([ np.cos(np.deg2rad(Azimuth_Z + Est_X_t_nparr[2])) , np.sin(np.deg2rad(Azimuth_Z + Est_X_t_nparr[2] ))])
    # Signs swapped in the Rotation matrix in a12 and a21 due to numpy issues -- Need to fix it
    Rot_Matrix = np.array( [ [np.cos(np.deg2rad(Est_X_t_nparr[2])) , np.sin(np.deg2rad(Est_X_t_nparr[2]))]  , [ -np.sin(np.deg2rad(Est_X_t_nparr[2])) , np.cos(np.deg2rad(Est_X_t_nparr[2])) ]]).T
    MeasGlobal_Z_t_Vec = Est_X_t_nparr[:2].reshape(2,1) + (Rot_Matrix @ Sensor_Pose_inHost).reshape(2,-1) + Range_Z * Meas_loc_Vector
    
    Rot_Matrix_d_ori = np.array( [ [-np.sin(np.deg2rad(Est_X_t_nparr[2])) , -np.cos(np.deg2rad(Est_X_t_nparr[2]))]  , [ +np.cos(np.deg2rad(Est_X_t_nparr[2])) , -np.sin(np.deg2rad(Est_X_t_nparr[2])) ]]).T
    Meas_loc_Vector_d_ori = np.array([ -np.sin(np.deg2rad(Azimuth_Z + Est_X_t_nparr[2])) , np.cps(np.deg2rad(Azimuth_Z + Est_X_t_nparr[2] ))])
    MeasGlobal_d_d_ori_t_Vec = (Rot_Matrix_d_ori @ Sensor_Pose_inHost).reshape(2,-1) + Range_Z * Meas_loc_Vector_d_ori
    
    Closest_x , Closest_y = np.argmin(np.abs(np.subtract(MeasGlobal_Z_t_Vec,Map_X_t_1))) 
     
    Dist = ( (MeasGlobal_Z_t_Vec[0] - Closest_x )**2 ) + ((MeasGlobal_Z_t_Vec[1] - Closest_y )**2 )
    Dist_d_x = 2 *(MeasGlobal_Z_t_Vec[0] - Closest_x )
    Dist_d_y = 2 *(MeasGlobal_Z_t_Vec[1] - Closest_y )
    Dist_d_ori = 2* (((MeasGlobal_Z_t_Vec[0] - Closest_x )*MeasGlobal_d_d_ori_t_Vec[0]) + ((MeasGlobal_Z_t_Vec[1] - Closest_y )*MeasGlobal_d_d_ori_t_Vec[1]))
    
    a = z_hit * (2*pi*(sigma**2))**0.5
    b = -0.5 *Dist/(sigma**2)
    c = (1 - z_hit)/o_max
    
    db_d_x = -Dist_d_x /(2*(sigma**2))
    db_d_y = -Dist_d_y /(2*(sigma**2))
    db_d_ori = -Dist_d_ori /(2*(sigma**2))
    
    Norm_Distribution = a*np.exp(b)
    den = Norm_Distribution +c
    log_q = np.log(den)
    dlog_q_d_x = (Norm_Distribution/den) * db_d_x
    dlog_q_d_y = (Norm_Distribution/den) * db_d_y
    dlog_q_d_ori = (Norm_Distribution/den) * db_d_ori
    
    d_x += dlog_q_d_x
    d_x += dlog_q_d_x
    d_ori += dlog_q_d_ori

if __name__ =='__main__':
    Likelihood_Field_Observation_Model_derivative(data['Est_X_t'],data['Meas_Z_t'],data['Map_init'])
    