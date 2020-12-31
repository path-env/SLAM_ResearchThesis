 # -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 23:11:25 2020

@author: MangalDeep
"""

from MotionModels import Odometry_Motion_Model_Cont , Odometry_Motion_Model_Sample
from ObservationModels import Likelihood_Field_Observation_Model
import numpy as np
from OccupancyGrid import Map
import pandas as pd
from RemoveGroundPlane import RANSAC,Zbased
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from matplotlib import style
from  ScanMatching import ScanMatching_OG
import math

#style.use('fivethirtyeight')
class RBPF_SLAM():
    def __init__(self):
        # Parameters
        self.Particle_Cnt = 50
        #self.InitParticle_Cnt = 10
        self.iteration = 0
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)
        self.MapSize = 100  #Local Map size
        self.True_x = [0]
        self.True_y = [0]
        self.True_yaw = [0]
        self.SM = ScanMatching_OG()
        self.OG = Map()
        
    def GPS_2_XY(self):
        #In Meters
        lon2,lat2 = self.True_x[-1] , self.True_y[-1]
        lon1,lat1 =  self.True_x[-2] , self.True_y[-2]
        dx = (lon1-lon2)*40000*math.cos((lat1+lat2)*math.pi/360)/360
        dy = (lat1-lat2)*40000/360
        return dx*1000,dy*1000
        
    #Initalize the system
    def Initialize(self,GPS_Z_t,IMU_Z_t,Meas_Z_t):
        #One Particle Initialized --With GPS data?
        P_cov = [np.diag([1,1,1]).flatten()]
        GPS_Z_t['long'] = 0
        GPS_Z_t['lat'] = 0
        IMU_Z_t['yaw'] = 0
        Meas_X_t = {'x': GPS_Z_t['long'] ,'y':GPS_Z_t['lat'] ,'yaw': IMU_Z_t['yaw']}
        Local_Map = self.OG.Update(Meas_X_t,Meas_Z_t,True).flatten()
        self.SM.CreateNDMap(self.OG)
        #Initalize Particle_Cnt Particles
        self.Particle_DFrame = pd.DataFrame({'x':GPS_Z_t['long'] ,
                                             'y':GPS_Z_t['lat'] ,
                                             'yaw': IMU_Z_t['yaw'],
                                             'pos_cov': P_cov, 
                                             'map': [Local_Map],
                                             'meas_likly':1/self.Particle_Cnt,
                                             'parent':[[0]],
                                             'N_weight':1/self.Particle_Cnt},
                                            index=(range(self.Particle_Cnt)))
    
    def Sampling(self,Meas_X_t, Meas_Z_t):
        #Sample from motion model
        for i in range(len(self.Particle_DFrame)):                
            if self.Meas_X_t_1['t'] != Meas_X_t['t']:
                Est_X_t_1 = self.Particle_DFrame.iloc[i,:3].to_dict()
                Est_X_t = Odometry_Motion_Model_Sample(self.Meas_X_t_1, Meas_X_t, Est_X_t_1)
                self.Particle_DFrame.iloc[i,0] = Est_X_t['x']
                self.Particle_DFrame.iloc[i,1] = Est_X_t['y']
                self.Particle_DFrame.iloc[i,2] = Est_X_t['yaw']
            #Find Likelyhood            
            # self.Particle_DFrame.iloc[i,5] = Likelihood_Field_Observation_Model(Meas_Z_t,
            #                                 Est_X_t, self.Particle_DFrame.iloc[i,4],Est_X_t_1,self.OG_Map_obj)
            
            Est_X_t = Est_X_t_1
            #ScanMatch
            MSE = self.SM.Scan_2_Map(Meas_Z_t,self.Meas_Z_t_1,Est_X_t,Est_X_t_1)
            if MSE > 2:
                self.Particle_DFrame.iloc[i,5] = MSE

            #Test
            #self.Particle_DFrame.at[i,'map'] = self.OG.Update(self.Particle_DFrame.iloc[i,:3].to_dict(),Meas_Z_t,True).flatten()
            
        #particle with the highest weight to update Map
        self.OG.UpdateMap()
        
        #Create a ND Map for Scan Match
        self.SM.CreateNDMap()
        
        #Normalize
        self.Particle_DFrame['meas_likly'] = self.Particle_DFrame['meas_likly'] / (self.Particle_DFrame['meas_likly'].sum())
        
        self.Importance_Weighting()            
    
    def Importance_Weighting(self):
        # Weights also need to be normalized
        self.Particle_DFrame['N_weight'] = self.Particle_DFrame['N_weight'] * self.Particle_DFrame['meas_likly']
        
        #Normalize
        self.Particle_DFrame['N_weight'] = self.Particle_DFrame['N_weight'] / (self.Particle_DFrame['N_weight'].sum())
        
        #Check if resampling is required        
        Den = self.Particle_DFrame['N_weight']**2
        N_eff = 1/ Den.sum()
        if N_eff < (self.Particle_Cnt/2):
            self.SysResample()
        else:
            print('Sampling Not required')    
        
    def SysResample(self):
        #Systematic Resampling
        # Effective set of particles N_eff to estimate how wel the current particle set represents the true posterior
        CDF = []
        CDF.append(0)
        for i in range(1,len(self.Particle_DFrame)):
            CDF.append(CDF[i-1] + self.Particle_DFrame.at[i,'N_weight'])
        
        Sample = []
        Sample.append(np.random.uniform(0,1/self.Particle_Cnt))
        i=0
        for j in range(self.Particle_Cnt):
            Sample.append(Sample[0] + ((j-1)/ self.Particle_Cnt))
            if i >= self.Particle_Cnt:
                Warning.message('Warn')
                break
            while Sample[j] > CDF[i]:
                i = i+1
            self.Particle_DFrame.at[j,'x'] = self.Particle_DFrame.at[i,'x']
            self.Particle_DFrame.at[j,'y'] = self.Particle_DFrame.at[i,'y']
            self.Particle_DFrame.at[j,'yaw'] = self.Particle_DFrame.at[i,'yaw']
            self.Particle_DFrame.at[j,'map'] = self.Particle_DFrame.at[i,'map']
            self.Particle_DFrame.at[j,'meas_likly'] = self.Particle_DFrame.at[i,'meas_likly']
            self.Particle_DFrame.at[j,'N_weight'] =  (1/self.Particle_Cnt)
            self.Particle_DFrame.at[j,'parent'] = self.Particle_DFrame.at[j,'parent'].append(i)
    
    def RandomResample(self):
        #Sample with replacement w.r.t weights
        Choice_Indx = np.random.choice(np.arange(self.Particle_Cnt), self.Particle_Cnt,replace = True,p=self.Particle_DFrame['N_weight'])
        #New ParticleFrame
        Temp = self.Particle_DFrame.loc[Choice_Indx,:]
        self.Particle_DFrame = Temp.copy()
    
    def Run(self,Meas_X_t, Meas_Z_t, GPS_Z_t,IMU_Z_t):
        self.GT(GPS_Z_t,IMU_Z_t)
        if 'Range_XY_plane' not in Meas_Z_t.keys():
            Meas_Z_t = self.OG.Lidar_3D_Preprocessing(Meas_Z_t)
        if self.iteration ==0:
            self.Initialize(GPS_Z_t, IMU_Z_t, Meas_Z_t)            
            self.iteration +=1
            self.Meas_X_t_1 = Meas_X_t.copy()
            self.Meas_Z_t_1 = Meas_Z_t.copy() 
            return None
        self.Sampling(Meas_X_t, Meas_Z_t)
        self.Meas_X_t_1 = Meas_X_t.copy()
        self.Meas_Z_t_1 = Meas_Z_t.copy()
        self.iteration +=1
        
    def GT(self,GPS_Z_t,IMU_Z_t):
        self.True_x.append(GPS_Z_t['long'])
        self.True_y.append(GPS_Z_t['lat'])
        self.True_yaw.append(IMU_Z_t['yaw'])
        # plt.plot(self.True_x,self.True_y)
        # plt.show()
        # self.ax1.scatter(self.Particle_DFrame['x'],self.Particle_DFrame['y'])
        
if __name__ == "__main__":
    pass
