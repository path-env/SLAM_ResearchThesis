# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:02:41 2020

@author: MangalDeep
"""
# To resolve VS code error
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
import pandas as pd

from libs.motion_models import Odometry_Motion_Model_Cont , Odometry_Motion_Model_Sample
from libs.occupancy_grid import Map

class FastSLAM2():
    def __init__(self):
        # Parameters
        self.Particle_Cnt = 100
        self.InitParticle_Cnt = 10
        self.Samples_Cnt =100
        self.PoseSmplPnts = []
        self.MapSize = 100  #Local Map size
        '''
        # Particle Template
        self.Particles_DFrame = pd.DataFrame({'x':[0],
                                            'y':[0],
                                            'yaw': [0], 
                                            'pos_cov': [0], 
                                            'map': [np.zeros((self.MapSize,))], 
                                            'N_weight':1/self.Particle_Cnt, 
                                            'mu':[0] , 
                                            'sigma':[0]},index=(range(self.Particle_Cnt)))
        '''
        self.Initialize()
        
    #Initalize the system
    #Initial Pose obtained with the Inital variance - Construct a Gausian - Draw N samples from it
    def Initialize(self):
        #One Particle Initialized --With GPS data?
        P_cov = 0#[np.diag([1,1,1]).flatten()]
        GPS_Z_t = {}
        IMU_Z_t = {}
        GPS_Z_t['x'] = 10
        GPS_Z_t['y'] = 10
        IMU_Z_t['yaw'] = 0
        #Init_map = Map()
        Local_Map = 0#Init_map.Update(Pose_X_t,Meas_Z_t,count).flatten()
        #Initalize N Particles
        self.Particle_DFrame = pd.DataFrame({'x':GPS_Z_t['x'] ,
                                             'y':GPS_Z_t['y'] ,
                                             'yaw': IMU_Z_t['yaw'],
                                             'mu':0,
                                             'norm': 0,
                                             'cov': P_cov, 
                                             'map': [Local_Map], 
                                             'N_weight':1/self.Particle_Cnt},
                                            index=(range(self.Particle_Cnt)))
        
        #Add Gaussian Noise to states individually
        self.Particle_DFrame['x'] = self.Particle_DFrame['x'] + np.random.randn(100)
        self.Particle_DFrame['y'] = self.Particle_DFrame['y'] + np.random.randn(100)
        self.Particle_DFrame['yaw'] = self.Particle_DFrame['yaw'] + np.random.randn(100)        
        
        '''
        #self.Particle_Tmplt = {'x':[] ,'y':[] ,'yaw': [], 'pos_cov': [], 'map': [], 'Nweight':0 , 'mu':0 , 'sigma':[]}
        #self.Particle_Lst = []
        
        #Preset Methods
        self.Generate_Particles()
        #Generate Particles  --- > Not required
    def Generate_Particles(self):
        self.Particle_set = pd.concat([self.Particle_Tmplt]*self.Particle_Cnt, ignore_index=True)
        

     '''    
    
    # p(x_t | x_t_1_i ,m_i,z_t,u_t ) = p(z_t|m_t_1,x_t) / Int(p(z_t|m_t_1,x_t))
    # Apply ICP to find the X with the least error and sample points in its region and asssign weights
    # Sampling from Proposal Distribution
    def Sampling(self,ParticleNo,Meas_X_t_1,Meas_X_t,Meas_Z_t,Map_t_1):
        #Sample from Odometry model for `scan matching
        Error_min = 1
        threshold = 0.02
        Est_X = Odometry_Motion_Model_Sample(Meas_X_t_1, Meas_X_t ,Updtd_X_t_1)
        #Apply ICP for scan matching
        Error = o3d.pipelines.registration.evaluate_registration(source, target, threshold)
        if Error_min > Error: #failure
            Est_X_t = Odometry_Motion_Model_Sample(Meas_X_t_1, Meas_X_t ,Updtd_X_t_1)
            
            self.Particle_DFrame[i,'N_weight'] = self.Particle_DFrame[i,'N_weight'] * Likelihood_Field_Observation_Model(Meas_Z_t , Est_X_t , Map_X_t_1)
        else:
            #Sample around Mode of the Estimated Proposal
            for SmplNo in self.Samples_Cnt:
                Sampled_Pose = Est_X + np.random.randn(0.02)
                
                temp = Likelihood_Field_Observation_Model(Meas_Z_t , Sampled_Pose , Map_X_t_1) @    Odometry_Motion_Model_Sample(Meas_X_t_1, Meas_X_t ,Updtd_X_t_1)
                #Compute Mu 
                self.Particle_DFrame[i,'mu'] =self.Particle_DFrame[i,'mu'] + Sampled_Pose @ temp 
                
                #Compute Norm 
                self.Particle_DFrame[i,'norm'] = self.Particle_DFrame[i,'norm'] + temp
                
                self.Particle_DFrame[i,'mu'] = self.Particle_DFrame[i,'mu']/self.Particle_DFrame[i,'norm']
                
                #Compute Cov
                MeanDiff = Sampled_Pose - self.Particle_DFrame[i,'mu']
                self.Particle_DFrame[i,'cov'] = self.Particle_DFrame[i,'cov'] + MeanDiff @ MeanDiff.T @ temp
                
                self.Particle_DFrame[i,'cov'] = self.Particle_DFrame[i,'cov']/self.Particle_DFrame[i,'norm'] 
            
            #Sample New Pose
            Est_x = self.Particle_DFrame[i,'mu'] + np.random.randn(self.Particle_DFrame[i,'cov'])
            
            self.Particle_DFrame[i,'N_weight'] = self.Particle_DFrame[i,'N_weight'] *self.Particle_DFrame[i,'norm']

    def Importance_Weighting(self,Normalizer_Lst):
        # Weights also need to be normalized
        Normalizer_Lst = Normalizer_Lst/np.sum(Normalizer_Lst)
        self.Particle_DFrame['N_weight'] = self.Particle_DFrame['N_weight'] * Normalizer_Lst
            
    def Resampling(self):
        # Effictive set of particles N_eff to estimate how wel the current particle set represents the true posterior
        Den = self.Particle_Lst['N_weight']**2
        N_eff = 1/ Den.sum()
        if N_eff < (self.Particle_Cnt/2):
            self.Sampling(Meas_X_t_1, Meas_X_t, Meas_Z_t, Map)
        else:
            print('Sampling Not required')
    
    def run(self):
        for ParticleNo in (self.Particle_Cnt):
            Meas_X_t_1
            Meas_X_t
            Updtd_X_t_1
            self.Sampling(ParticleNo,Meas_X_t_1,Meas_X_t,Meas_Z_t,Map_t_1)
            
            
            
            
if __name__ == "__main__":
    Obj = FastSLAM2()
    