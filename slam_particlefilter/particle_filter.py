# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 23:11:25 2020

@author: MangalDeep
"""
import numpy as np
import pandas as pd

import math
import logging

from libs.occupancy_grid import Map
from libs.motion_models import CTRA_Motion_Model, VMM
from libs.scan_matching import ICP, RTCSM
from utils.tools import Lidar_3D_Preprocessing
from libs.slam_analyzer import Analyze as aly

class RBPF_SLAM():
    def __init__(self, plotter, N):
        # Parameters
        self.Particle_Cnt = N
        self.iteration = 0
        self.MapSize = 100  # Local Map siz
        self.logger = logging.getLogger('ROS_Decode.RBPF_SLAM')
        self.logger.info('creating an instance of RBPF_SLAM')
        self.prev_scan_update = {'x': 0, 'y': 0, 'yaw': 0}
        self.best_particle = 0
        self.WhlWidth = 0.8568637084960926*2 #type"vehicle.mini.cooperst" Wheels.position.x *2
        self.WHlBase = (1.143818969726567 + 1.367382202148434)/2 #type"vehicle.mini.cooperst" Wheels.position.y *2
        self.TyrRds = 37.5/1000 #m
        self.aly = plotter

    # Initalize the system
    def _initialize(self, GPS_Z_t, IMU_Z_t, Meas_Z_t, Meas_X_t):
        self.logger.info('Initializing PF values')
        # One Particle initialized --With GPS data?
        GPS_Z_t['long'] = Meas_X_t['x']
        GPS_Z_t['lat'] = Meas_X_t['y']
        IMU_Z_t['yaw'] = Meas_X_t['yaw']
        #Meas_X_t = {'x': Meas_X_t['x'], 'y': Meas_X_t['y'], 'yaw': Meas_X_t['yaw']}
       # Local_Map = self.OG.Update(Meas_X_t, Meas_Z_t, True).flatten()
        #self.SM.CreateNDMap(self.OG, Meas_X_t)
        # Initalize Particle_Cnt Particles
        st = [GPS_Z_t['long'],  GPS_Z_t['lat'], IMU_Z_t['yaw'], Meas_X_t['v']]# x,y,yaw,v  
        self.Particle_DFrame = pd.DataFrame({'st':[st], 
                                             'map': [0],
                                             'parent': np.arange(self.Particle_Cnt),
                                             'N_weight': 1},
                                            index=(range(self.Particle_Cnt)))
        #Noise_Mtrx = np.hstack((np.random.randn(10, 1) ,np.random.randn(10, 1) ,np.random.randn(10, 1) ,np.zeros((10,1))))
        #self.Particle_DFrame['st'] = self.Particle_DFrame['st'].to_list() + Noise_Mtrx
        self.Particle_DFrame =  self.Particle_DFrame.assign(meas_likly = [1/self.Particle_Cnt for i in range(self.Particle_Cnt)])
        # self.Particle_DFrame =  self.Particle_DFrame.assign(traject_x = [[GPS_Z_t['long']] for i in range(self.Particle_Cnt)])
        # self.Particle_DFrame =  self.Particle_DFrame.assign(traject_y = [[GPS_Z_t['lat']] for i in range(self.Particle_Cnt)])
        # self.Particle_DFrame =  self.Particle_DFrame.assign(traject_yaw = [[IMU_Z_t['yaw']] for i in range(self.Particle_Cnt)])

    def _sampling(self, Meas_X_t, Meas_Z_t,IMU_Z_t):
        self.logger.info('sampling from Motion Model')
        # Sample from motion model
        if self.Meas_X_t_1['t'] != Meas_X_t['t']:
            if not(Meas_X_t['v']>0.01 or Meas_X_t['v'] <-0.01):
                #print(f"Vehicle stationary at @ {Meas_X_t['t']}")
                self.Particle_DFrame['v'] = 0
                #self._importance_weighting(Meas_Z_t)
                return None 
            Meas_X_t['yaw_dot'] = IMU_Z_t['ang_vel']
            cmdIn = np.array([Meas_X_t['yaw_dot'], Meas_X_t['acc'], self.Meas_X_t_1['acc']]).reshape(3,)
            dt=Meas_X_t['t'] - self.Meas_X_t_1['t']
            for i in range(len(self.Particle_DFrame)):
                Est_X_t_1 = self.Particle_DFrame.at[i, 'st']
                # Est_X_t = Odometry_Motion_Model_Sample(self.Meas_X_t_1, Meas_X_t, Est_X_t_1)
                #Meas_X_t['yaw_dot'] = Meas_X_t['v'] * (np.tan(Meas_X_t['steer'])/self.WHlBase)
                Est_X_t = CTRA_Motion_Model(Est_X_t_1, cmdIn, dt)

                self.Particle_DFrame.at[i, 'st'] = Est_X_t
                
                #Est = {'x':0, 'y':0,'yaw':0}
                 # ScanMatch - RTCSM
                #self.SM.match(Est_X_t, Meas_Z_t)
                 # ScanMatch - ICP
                GlobalTrans = self.SM.match(Meas_Z_t.to_numpy().T, 
                                                      self.Meas_Z_t_1.to_numpy().T, 
                                                      self.Particle_DFrame.at[i, 'st'], 
                                                      Est_X_t_1)
                
                # Error Beyond bound
                # if GlobalTrans['error'] >5:
                #     print(f"Skipping Particle update for {i}, and error:{GlobalTrans['error']} ")
                #     continue

                #print(translation, orientation,error)
                self.Particle_DFrame.at[i, 'meas_likly'] = GlobalTrans['error']

            self._importance_weighting(Meas_Z_t)
        
    def _importance_weighting(self, Meas_Z_t):
        self.logger.info('Importance Weighting')
        self.Particle_DFrame['N_weight'] = self.Particle_DFrame['N_weight'] *(1- self.Particle_DFrame['meas_likly'])       
        # Weights  need to be normalized
        # Apply softmax to handle negative weights
        self.Particle_DFrame['N_weight'] = np.exp(self.Particle_DFrame['N_weight'])/ \
                                            np.sum(np.exp(self.Particle_DFrame['N_weight']))

        # Check if resampling is required
        Den = self.Particle_DFrame['N_weight'] ** 2
        N_eff = 1 / Den.sum()
        if np.any(self.Particle_DFrame['N_weight']==np.nan):
            print(self.Particle_DFrame['N_weight'] )
        if N_eff < (self.Particle_Cnt / 2):
            self._random_resample()
        else:
            self.logger.info('Resampling Not required')

        #Set estimate to the mean value
        self._est_state() 

    def _est_state(self):
        # weighed sum of each particle
        mean = np.average(self.Particle_DFrame['st']*self.Particle_DFrame['N_weight'], axis=0)*10
        pred_mat = np.concatenate( self.Particle_DFrame['st'],axis = 0).reshape(10,4) 
        prod = np.multiply(((pred_mat- mean)**2).T,self.Particle_DFrame['N_weight'].to_numpy())
        var =  np.average(prod)

        # particle with the highest weight to update Map
        #self.best_particle = self.Particle_DFrame.meas_likly.idxmin()
        Est_X_t = mean #self.Particle_DFrame.loc[self.best_particle].to_dict()       
        self.prev_scan_update = Est_X_t.copy()
        self.aly._set_trajectory(Est_X_t)
        #self.OG.Update(Est_X_t, Meas_Z_t, True)

    def _sys_resample(self):
        self.logger.info("Systematic resampling....")
        # Systematic Resampling
        # Effective set of particles N_eff to estimate how wel the current particle set represents the true posterior
        CDF = []
        CDF.append(0)
        for i in range(1, len(self.Particle_DFrame)):
            CDF.append(CDF[i - 1] + self.Particle_DFrame.at[i, 'N_weight'])

        Sample = []
        Sample.append(np.random.uniform(0, 1 / self.Particle_Cnt))
        i = 0
        for j in range(self.Particle_Cnt):
            Sample.append(Sample[0] + ((j - 1) / self.Particle_Cnt))
            if i >= self.Particle_Cnt:
                Warning.message('Warn')
                break
            while Sample[j] > CDF[i]:
                i = i + 1
            self.Particle_DFrame.at[j, 'x'] = self.Particle_DFrame.at[i, 'x']
            self.Particle_DFrame.at[j, 'y'] = self.Particle_DFrame.at[i, 'y']
            self.Particle_DFrame.at[j, 'yaw'] = self.Particle_DFrame.at[i, 'yaw']
            self.Particle_DFrame.at[j, 'map'] = self.Particle_DFrame.at[i, 'map']
            self.Particle_DFrame.at[j, 'meas_likly'] = self.Particle_DFrame.at[i, 'meas_likly']
            self.Particle_DFrame.at[j, 'N_weight'] = (1 / self.Particle_Cnt)
            self.Particle_DFrame.at[j, 'parent'] = self.Particle_DFrame.at[j, 'parent'].append(i)

    def _random_resample(self):
        # Sample with replacement w.r.t weights
        self.logger.info("Random resampling....")
        Choice_Indx = np.random.choice(np.arange(self.Particle_Cnt), self.Particle_Cnt, replace=True,
                                       p=self.Particle_DFrame['N_weight'])
        # New ParticleFrame
        Temp = self.Particle_DFrame.loc[Choice_Indx, :].reset_index(drop=True)
        self.Particle_DFrame = Temp.copy()
        self.Particle_DFrame['N_weight'] = 1 / self.Particle_Cnt

    def run(self, Meas_X_t, Meas_Z_t, GPS_Z_t, IMU_Z_t):
            # try:       
        self.logger.info('RBPF_SLAM_Iteration.......%d for time = %f(s)', self.iteration, Meas_X_t['t'])
        
        if 'Range_XY_plane' not in Meas_Z_t.keys():
            Meas_Z_t = Lidar_3D_Preprocessing(Meas_Z_t)
        if self.iteration == 0:
            self.OG = Map()
            self.SM = ICP()
            #self.SM = RTCSM(self.OG, Meas_X_t, Meas_Z_t)
            self.Meas_X_t_1 = Meas_X_t.copy()
            self._initialize(GPS_Z_t, IMU_Z_t, Meas_Z_t,Meas_X_t)
            self.iteration += 1
            self.Meas_Z_t_1 = Meas_Z_t.copy()
            return None
        # Check Timeliness
        assert (self.Meas_X_t_1['t'] - Meas_X_t['t']) <= 1, "Time difference is very high"
        self._sampling(Meas_X_t, Meas_Z_t, IMU_Z_t)
        self.Meas_X_t_1 = Meas_X_t.copy()
        self.Meas_Z_t_1 = Meas_Z_t.copy()
        self.iteration += 1
                
            # except Exception as e:
            #     print(e)
            #     raise(RuntimeWarning)
            #finally:
                
        self.aly.plot_results() 

    def _gps_2_XY(self):
        # In Meters
        lon2, lat2 = self.aly.True__x[0], self.aly.True__y[0]
        lon1, lat1 = self.aly.True__x[-2], self.aly.True__y[-2]
        dx = (lon1 - lon2) * 40000 * math.cos((lat1 + lat2) * math.pi / 360) / 360
        dy = (lat1 - lat2) * 40000 / 360
        return dx * 1000, dy * 1000
    
         
if __name__ == '__main__':
    from main.ROSBag_decode import ROS_bag_run
    ROS_bag_run()