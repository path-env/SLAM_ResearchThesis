# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:49:57 2020

@author: MangalDeep
"""

import random
import numpy as np
import sympy as sp
from sympy import sin, cos, Matrix,atan2,deg
import pickle 
import tarfile # open a .spydata file filename =

#################################### Sample Based Models ############################################
#Odometry(Sampling)
def Odometry_Motion_Model_Sample(Meas_X_t_1, Meas_X_t , Est_X_t_1):
    #Est Init
    Est_X_t = {}
    #Params
    a1 = 0.05   #m/m
    a2 = 0.001  #m/deg
    a3 = 0.05      #deg/m
    a4 = 0.05   #deg/deg
    
    Del_rot1 = np.rad2deg( np.arctan2( (Meas_X_t['y'] - Meas_X_t_1['y']) , (Meas_X_t['x'] - Meas_X_t_1['x']) ) )-Meas_X_t_1['yaw']
    Del_trans = np.hypot( ((Meas_X_t_1['y'] - Meas_X_t['y'])) , (Meas_X_t_1['x'] - Meas_X_t['x']) )
    Del_rot2 = Meas_X_t['yaw'] - Meas_X_t_1['yaw'] - Del_rot1
    
    Del_est_rot1  = Del_rot1 -   Sample_Gaus_dist( a1*Del_rot1**2  + a2*Del_trans**2 )
    Del_est_trans  = Del_trans - Sample_Gaus_dist( a3*Del_trans**2 + a4*Del_rot1**2 + a4*Del_rot2**2 )
    Del_est_rot2 = Del_rot2 -    Sample_Gaus_dist( a1*Del_rot2**2  + a2*Del_trans**2 )
    
    Est_X_t['x'] = Est_X_t_1['x'] + Del_est_trans*np.cos( np.deg2rad(Est_X_t_1['yaw'] + Del_est_rot1 ))
    Est_X_t['y'] = Est_X_t_1['y'] + Del_est_trans*np.sin( np.deg2rad(Est_X_t_1['yaw'] + Del_est_rot1 ))
    Est_X_t['yaw'] = Est_X_t_1['yaw'] + Del_est_rot1 + Del_est_rot2 
    return Est_X_t


def Sample_Gaus_dist(val):
    return (val/6)*np.sum([random.uniform(-1,1) for i in range(12)])

def Sample_Triangular_dist(val):
    return val*random.uniform(-1,1)*random.uniform(-1,1)

#################################### Continuous Models ############################################

def Odometry_Motion_Model_Cont(Meas_X_t_1, Meas_X_t  ,Hypth_X_t,Updtd_X_t_1):
    #Params
    a1 = 0.01
    a2 = 0.02
    a3 = 0.01
    a4 = 0.01
    
    Del_rot1 = np.rad2deg( np.arctan2( (Meas_X_t['y'] - Meas_X_t_1['y']) , (Meas_X_t['x'] - Meas_X_t_1['x']) ) )-Meas_X_t_1['yaw']
    Del_trans = np.hypot( ((Meas_X_t_1['y'] - Meas_X_t['y'])) , (Meas_X_t_1['x'] - Meas_X_t['x']) )
    Del_rot2 = Meas_X_t['yaw'] - Meas_X_t_1['yaw'] - Del_rot1
    
    Del_est_rot1  = np.rad2deg( np.arctan2( (Updtd_X_t_1['y'] - Hypth_X_t['y']) , (Updtd_X_t_1['x'] - Hypth_X_t['x']) ))
    Del_est_trans  = np.hypot( ((Updtd_X_t_1['y'] - Hypth_X_t['y'])) , (Updtd_X_t_1['x'] - Hypth_X_t['x']) )
    Del_est_rot2 = Updtd_X_t_1['yaw'] - Hypth_X_t['yaw'] - Del_est_rot1
    
    P1 = Prob_Gaus_dist( (Del_rot1 - Del_est_rot1) , (a1*Del_est_rot1 + a2*Del_est_trans) )
    P2 = Prob_Gaus_dist( (Del_trans - Del_est_trans) , (a3*Del_est_trans + a4*(Del_est_rot1+Del_est_rot2)) )
    P3 = Prob_Gaus_dist( (Del_rot2 - Del_est_rot2) , (a1*Del_est_rot2 + a2*Del_est_trans) )
    
    return P1*P2*P3


def Prob_Gaus_dist(MeanCentered , Var):
    return ((np.sqrt(2*np.pi*Var)**-1) *np.exp( (-0.5* MeanCentered**2)/Var))

def Prob_Triangular_dist(MeanCentered , Var):
    if np.abs(MeanCentered) > np.sqrt(6*Var):
        return 0
    else:
        return ((np.sqrt(6*Var) - np.abs(MeanCentered)) / (6*Var))
    
#################################### Bicycle Model Continuous ################################################
#Bicycle Model with velocity vector at the rear wheel. Only Longitudinal Kinematics considered
def BicycleModel(Meas_X_t_1,In_X_t, dt , L = 4.7):
    Est_X_t = Meas_X_t_1
    Est_X_t['v'] = Meas_X_t_1['v'] + In_X_t['acc'] *dt    
    Est_X_t['x'] = Meas_X_t_1['x'] +  Est_X_t['v']*np.cos(np.deg2rad(Meas_X_t_1['yaw'])) *dt
    Est_X_t['y'] = Meas_X_t_1['y'] +  Est_X_t['v']*np.sin(np.deg2rad(Meas_X_t_1['yaw'])) *dt
    Est_X_t['yaw'] = Meas_X_t_1['yaw'] + ( Est_X_t['v']*np.tan(np.deg2rad(In_X_t['steer'])) *dt /L)
    return Est_X_t

#################################### Bicycle Model Discretized ################################################
#Bicycle Model with velocity vector at the rear wheel. Only Longitudinal Kinematics considered
def BicycleModel(Meas_X_k_1,In_X_k, dt , L = 4.7):
    Est_X_k = Meas_X_k_1
    Est_X_k['v'] = Meas_X_k_1['v'] + In_X_k['acc'] *dt    
    Est_X_k['x'] = Meas_X_k_1['x'] +  Est_X_k['v']*np.cos(np.deg2rad(Meas_X_k_1['yaw'])) *dt
    Est_X_k['y'] = Meas_X_k_1['y'] +  Est_X_k['v']*np.sin(np.deg2rad(Meas_X_k_1['yaw'])) *dt
    Est_X_k['yaw'] = Meas_X_k_1['yaw'] + ( Est_X_k['v']*np.tan(np.deg2rad(In_X_k['steer'])) *dt /L)
    return Est_X_k

################################### CTRA Motion Model Discretized #############################################

def CTRA_Motion_Model(Meas_X_k_1, cmdIn, WheelBase,dt,y_dot_tolerance=1):
    #Params
    a1 = 0.05   #m/m
    a2 = 0.001  #m/deg
    a3 = 0.05      #deg/m
    a4 = 0.05   #deg/deg
    Est_X_k = {}
    
    
    del_v =  (cmdIn[2]-cmdIn[1]) *(dt**2)/2
    Est_X_k['v'] = Meas_X_k_1['v'] + del_v -Sample_Gaus_dist( a1* cmdIn[0]**2  + a2*cmdIn[1]**2 )
    
    #del_yawdot = Meas_X_k_1['v'] * np.tan(steer) *dt /WheelBase
    # del_yawdot = dt*cmdIn[0]
    # Meas_X_k_1['yaw_dot'] = cmdIn[0]
    # Est_X_k['yaw_dot']= cmdIn[0] -del_yawdot
    
    del_yaw =  cmdIn[0] *dt
    Est_X_k['yaw'] =  np.deg2rad(Meas_X_k_1['yaw']) - del_yaw
    Meas_X_k_1['yaw'] = np.deg2rad(Meas_X_k_1['yaw'])
    if  cmdIn[0] > y_dot_tolerance:
        # del_x = Est_X_k['v']*(np.sin(Est_X_k['yaw'])/Est_X_k['yaw_dot']) -\
        #         Meas_X_k_1['v']*(np.sin(Meas_X_k_1['yaw'])/Meas_X_k_1['yaw_dot']) +\
        #         cmdIn[1]*((np.cos(Est_X_k['yaw'])/Est_X_k['yaw_dot']**2) -(np.cos(Meas_X_k_1['yaw'])/Meas_X_k_1['yaw_dot']**2))
                
        # del_y = -Est_X_k['v']*(np.cos(Est_X_k['yaw'])/Est_X_k['yaw_dot']) +\
        #         Meas_X_k_1['v']*(np.cos(Meas_X_k_1['yaw'])/Meas_X_k_1['yaw_dot']) +\
        #         cmdIn[1]*((np.sin(Est_X_k['yaw'])/Est_X_k['yaw_dot']**2) -(np.sin(Meas_X_k_1['yaw'])/Meas_X_k_1['yaw_dot']**2))
                
        del_x = ( (Est_X_k['v']* cmdIn[0]*np.sin(Est_X_k['yaw'])) + (cmdIn[1] *np.cos(Est_X_k['yaw']))-
                                          (Meas_X_k_1['v']* cmdIn[0]*np.sin(Meas_X_k_1['yaw'])) - 
                                          (cmdIn[1]*np.cos(Meas_X_k_1['yaw'])))/( cmdIn[0]**2) 
    
        del_y = ( -(Est_X_k['v']* cmdIn[0]*np.cos(Est_X_k['yaw'])) + (cmdIn[1] *np.sin(Est_X_k['yaw']))+
                                          (Meas_X_k_1['v']* cmdIn[0]*np.cos(Meas_X_k_1['yaw'])) - 
                                          (cmdIn[1] *np.sin(Meas_X_k_1['yaw'])))/( cmdIn[0]**2)    
    else:
        del_x = (Meas_X_k_1['v']*dt + (cmdIn[1] *(dt**2))/2 )*np.cos(Meas_X_k_1['yaw'])
        
        del_y = (Meas_X_k_1['v']*dt + (cmdIn[1]  *(dt**2))/2 )*np.sin(Meas_X_k_1['yaw'])
        
    Est_X_k['x'] =      Meas_X_k_1['x'] + del_x -  Sample_Gaus_dist( a1* cmdIn[0]**2  + a2*cmdIn[1]**2 )   
    Est_X_k['y'] =      Meas_X_k_1['y'] + del_y - Sample_Gaus_dist( a1* cmdIn[0]**2  + a2*cmdIn[1]**2 )
    
    Est_X_k['acc'] =    cmdIn[1]

    # xMove, yMove = Est_X_k['x'] - Meas_X_k_1['x'], Est_X_k['y'] - Meas_X_k_1['y']
    # move = np.sqrt(xMove ** 2 + yMove ** 2)
    # if move != 0:
    #     if yMove > 0:
    #         Est_X_k['yaw'] = np.arccos(xMove / move)
    #     else:
    #         Est_X_k['yaw'] = -np.arccos(xMove / move)
    # else:
    #     Est_X_k['yaw'] = None  
    
    Est_X_k['yaw'] =    np.rad2deg(Est_X_k['yaw'])-Sample_Gaus_dist( a3* cmdIn[0]**2  + a4*cmdIn[1]**2 )
    
    # prvSt = np.zeros((6,1))
    # prvSt[0:4,:] = np.array([*Meas_X_k_1.values()][:4]).reshape(4,1) #x,y,yaw,v
    # G = np.zeros((6,2))
    # G[4,0], G[5,1] = dt,dt
    # cmdIn = cmdIn # - np.array([Sample_Gaus_dist(a1*cmdIn[0]**2 + a2*cmdIn[1]**2), Sample_Gaus_dist(a1*cmdIn[0]**2  + a2*cmdIn[1]**2 )]).rehsape(2,1)
    # f_CTRA = np.zeros((6,1))
    # f_CTRA[0:4,0] = del_x, del_y, del_yaw, del_v#, del_yawdot
    
    # Est = prvSt + f_CTRA  + G @ cmdIn 
    return Est_X_k

def VMM(Meas_X_k_1, cmdIn,dt):
    a1 = 0.05  
    a2 = 0.05  
    a3 = 0.05      #deg/m
    a4 = 0.05   #deg/deg
    a5,a6 = 0.001,0.001
    v_hat = cmdIn[1] + Sample_Gaus_dist(a1 * cmdIn[0]**2 + a2*cmdIn[1]**2)
    w_hat = cmdIn[0] + Sample_Gaus_dist(a3 * cmdIn[0]**2 + a4*cmdIn[1]**2)
    crct = Sample_Gaus_dist(a5*cmdIn[0]**2 + a6*cmdIn[1]**2 )
    Est_X_k = {}
    Est_X_k['x'] = Meas_X_k_1['x'] - (v_hat/w_hat)*np.sin(Meas_X_k_1['yaw']) + (v_hat/w_hat)*np.sin(Meas_X_k_1['yaw'] + w_hat*dt)
    Est_X_k['y'] = Meas_X_k_1['y'] + (v_hat/w_hat)*np.cos(Meas_X_k_1['yaw']) - (v_hat/w_hat)*np.cos(Meas_X_k_1['yaw'] + w_hat*dt)
    Est_X_k['yaw'] = Meas_X_k_1['yaw']  + w_hat*dt + crct*dt
    return Est_X_k

if __name__ == '__main__':
    filename = 'G:/Masters-FHD/Sem3/ResearchThesis/Trial3.spydata'
    tar = tarfile.open(filename, "r") 
    tar.extractall() 
    extracted_files = tar.getnames() 
    for f in extracted_files: 
        if f.endswith('.pickle'): 
            with open(f,'rb') as fdesc: 
                data = pickle.loads(fdesc.read())