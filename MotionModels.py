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
filename = 'G:/Masters-FHD/Sem3/ResearchThesis/Trial3.spydata'
tar = tarfile.open(filename, "r") 
tar.extractall() 
extracted_files = tar.getnames() 
for f in extracted_files: 
    if f.endswith('.pickle'): 
        with open(f,'rb') as fdesc: 
            data = pickle.loads(fdesc.read())
#################################### Sample Based Models ############################################
#Odometry(Sampling)
def Odometry_Motion_Model_Sample(Meas_X_t_1, Meas_X_t , Updtd_X_t_1):
    #Est Init
    Est_X_t = {}
    #Params
    a1 = 1
    a2 = 1
    a3 = 1
    a4 = 1
    
    Del_rot1 = np.rad2deg( np.arctan2( (Meas_X_t['y'] - Meas_X_t_1['y']) , (Meas_X_t['x'] - Meas_X_t_1['x']) ) )-Meas_X_t_1['yaw']
    Del_trans = np.hypot( ((Meas_X_t_1['y'] - Meas_X_t['y'])) , (Meas_X_t_1['x'] - Meas_X_t['x']) )
    Del_rot2 = Meas_X_t['yaw'] - Meas_X_t_1['yaw'] - Del_rot1
    
    Del_est_rot1  = Del_rot1 - Sample_Gaus_dist( a1*np.deg2rad(Del_rot1) + a2*Del_trans )
    Del_est_trans  = Del_trans - Sample_Gaus_dist( a3*Del_trans + a4*(np.deg2rad(Del_rot1+Del_rot2) ))
    Del_est_rot2 = Del_rot2 - Sample_Gaus_dist( a1*np.deg2rad(Del_rot2) + a2*Del_trans )
    
    Est_X_t['x'] = Updtd_X_t_1['x'] + Del_est_trans*np.cos( np.deg2rad(Updtd_X_t_1['yaw'] + Del_est_rot1 ))
    Est_X_t['y'] = Updtd_X_t_1['y'] + Del_est_trans*np.sin( np.deg2rad(Updtd_X_t_1['yaw'] + Del_est_rot1 ))
    Est_X_t['yaw'] = Updtd_X_t_1['yaw'] + Del_est_rot1 + Del_est_rot2 
    
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

def CTRA_Motion_Model(Meas_X_k_1, dt):
    Est_X_k = Meas_X_k_1
    Est_X_k['v'] =      Meas_X_k_1['v'] +  Meas_X_k_1['acc'] *dt
    Est_X_k['yaw'] =    Meas_X_k_1['yaw'] + Meas_X_k_1['yaw_dot'] *dt
    
    Est_X_k['x'] =      Meas_X_k_1['x'] + ( (Est_X_k['v']* Meas_X_k_1['yaw_dot']*np.sin(Est_X_k['yaw'])) + (Meas_X_k_1['acc']*np.cos(Est_X_k['yaw']))-
                                         (Meas_X_k_1['v']* Meas_X_k_1['yaw_dot']*np.sin(Meas_X_k_1['yaw'])) - (Meas_X_k_1['acc']*np.cos(Meas_X_k_1['yaw'])) /Meas_X_k_1['yaw_dot']**2)
    
    Est_X_k['y'] =      Meas_X_k_1['y'] +  ( -(Est_X_k['v']* Meas_X_k_1['yaw_dot']*np.cos(Est_X_k['yaw'])) + (Meas_X_k_1['acc']*np.sin(Est_X_k['yaw']))+
                                         (Meas_X_k_1['v']* Meas_X_k_1['yaw_dot']*np.cos(Meas_X_k_1['yaw'])) - (Meas_X_k_1['acc']*np.sin(Meas_X_k_1['yaw'])) /Meas_X_k_1['yaw_dot']**2)
    Est_X_k['yaw_dot']= Meas_X_k_1['yaw_dot'] 
    Est_X_k['acc'] =    Meas_X_k_1['acc'] 
    return Est_X_k