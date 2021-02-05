# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 23:11:25 2020

@author: MangalDeep
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.animation as
# from matplotlib.animation import FuncAnimation
# from matplotlib import style
import math
import logging

from libs.occupancy_grid import Map
from libs.motion_models import VMM, Odometry_Motion_Model_Sample, CTRA_Motion_Model
from libs.observation_models import Likelihood_Field_Observation_Model
from libs.scan_matching import ICP, RTCSM
from utils.tools import Lidar_3D_Preprocessing

plt.rcParams.update({'font.size': 7})

class RBPF_SLAM():
    def __init__(self):
        # Parameters
        self.Particle_Cnt = 10
        self.iteration = 0
        # self.fig = plt.figure()
        # self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.MapSize = 100  # Local Map siz
        self.logger = logging.getLogger('ROS_Decode.RBPF_SLAM')
        self.logger.info('creating an instance of RBPF_SLAM')
        self.prev_scan_update = {'x': 0, 'y': 0, 'yaw': 0}
        self.best_particle = 0
        self.predict_x,self.predict_y,self.predict_yaw = [], [], []
        self.corrected_x,self.corrected_y,self.corrected_yaw, self.corrected_v = [], [], [], []
        self.True_x,self.True_y, self.True_yaw, self.True_v, self.True_acc = [], [], [],[], []
        self.odom_x, self.odom_y, self.odom_yaw = [],[],[]
        self.WhlWidth = 0.8568637084960926*2 #type: "vehicle.mini.cooperst" Wheels.position.x *2
        self.WHlBase = (1.143818969726567 + 1.367382202148434)/2 #type: "vehicle.mini.cooperst" Wheels.position.y *2
        self.TyrRds = 37.5/1000 #m
        self.steer = []
        self.time = []
        
    # Initalize the system
    def _initialize(self, GPS_Z_t, IMU_Z_t, Meas_Z_t, Meas_X_t):
        self.logger.info('Initializing PF values')
        # One Particle initialized --With GPS data?
        GPS_Z_t['long'] = Meas_X_t['x']
        GPS_Z_t['lat'] = Meas_X_t['y']
        IMU_Z_t['yaw'] = Meas_X_t['yaw']
        Meas_X_t = {'x': Meas_X_t['x'], 'y': Meas_X_t['y'], 'yaw': Meas_X_t['yaw']}
       # Local_Map = self.OG.Update(Meas_X_t, Meas_Z_t, True).flatten()
        #self.SM.CreateNDMap(self.OG, Meas_X_t)
        # Initalize Particle_Cnt Particles
        self.Particle_DFrame = pd.DataFrame({'x': GPS_Z_t['long'],
                                             'y': GPS_Z_t['lat'],
                                             'yaw': IMU_Z_t['yaw'],
                                             'v': self.Meas_X_t_1['v'],
                                             'map': [0],
                                             'parent': np.arange(self.Particle_Cnt),
                                             'N_weight': 1},
                                            index=(range(self.Particle_Cnt)))
        
        self.Particle_DFrame['x'] = self.Particle_DFrame['x'] + 0.001*np.random.randn(self.Particle_Cnt,1).flatten()
        self.Particle_DFrame['y'] = self.Particle_DFrame['y'] + 0.001*np.random.randn(self.Particle_Cnt,1).flatten()
        self.Particle_DFrame['yaw'] = self.Particle_DFrame['yaw'] + 0.001*np.random.randn(self.Particle_Cnt,1).flatten()
        self.Particle_DFrame =  self.Particle_DFrame.assign(meas_likly = [1/self.Particle_Cnt for i in range(self.Particle_Cnt)])
        self.Particle_DFrame =  self.Particle_DFrame.assign(traject_x = [[GPS_Z_t['long']] for i in range(self.Particle_Cnt)])
        self.Particle_DFrame =  self.Particle_DFrame.assign(traject_y = [[GPS_Z_t['lat']] for i in range(self.Particle_Cnt)])
        self.Particle_DFrame =  self.Particle_DFrame.assign(traject_yaw = [[IMU_Z_t['yaw']] for i in range(self.Particle_Cnt)])

    def _sampling(self, Meas_X_t, Meas_Z_t,IMU_Z_t):
        self.logger.info('sampling from Motion Model')
        # Sample from motion model
        if self.Meas_X_t_1['t'] != Meas_X_t['t']:
            if not(Meas_X_t['v']>0.01 or Meas_X_t['v'] <-0.01):
                print(f"Vehicle stationary at @ {Meas_X_t['t']}")
                self._importance_weighting(Meas_Z_t)
                return None
            for i in range(len(self.Particle_DFrame)):
                Est_X_t_1 = self.Particle_DFrame.iloc[i, :].to_dict()
                # Est_X_t = Odometry_Motion_Model_Sample(self.Meas_X_t_1, Meas_X_t, Est_X_t_1)
                # Est_X_t['v'] = 0
                #Meas_X_t['yaw_dot'] = Meas_X_t['v'] * (np.tan(Meas_X_t['steer'])/self.WHlBase)
                Meas_X_t['yaw_dot'] = -1*IMU_Z_t['ang_vel']
                cmdIn = np.array([Meas_X_t['yaw_dot'], Meas_X_t['acc'], self.Meas_X_t_1['acc']]).reshape(3,)
                #Est_X_t = VMM(Est_X_t_1, cmdIn,dt=Meas_X_t['t'] - self.Meas_X_t_1['t']  )
                Est_X_t = CTRA_Motion_Model(Est_X_t_1, cmdIn,self.WHlBase, dt=Meas_X_t['t'] - self.Meas_X_t_1['t'])

                self.Particle_DFrame.at[i, 'x'] = Est_X_t['x']
                self.Particle_DFrame.at[i, 'y'] = Est_X_t['y']
                self.Particle_DFrame.at[i, 'yaw'] = Est_X_t['yaw'] # degrees
                self.Particle_DFrame.at[i, 'v'] = Est_X_t['v']
                
                #Est = {'x':0, 'y':0,'yaw':0}
                self.SM.match(Est_X_t, Meas_Z_t)
                 # ScanMatch - ICP
                _,RelativeTrans = self.SM.match(Meas_Z_t.to_numpy().T, 
                                                      self.Meas_Z_t_1.to_numpy().T, 
                                                      Est_X_t, 
                                                      Est_X_t_1)
                # ScanMatch - RTCSM 
                self.SM.match()
                # Error Beyond bound
                if RelativeTrans['error'] == None:
                    print(f"Skipping Particle update for {i}")
                    continue

                #print(translation, orientation,error)
                self.Particle_DFrame.at[i, 'meas_likly'] = RelativeTrans['error']
                self.Particle_DFrame.at[i, 'traject_x'].append(RelativeTrans['t'][0])
                self.Particle_DFrame.at[i, 'traject_y'].append(RelativeTrans['t'][1])
                self.Particle_DFrame.at[i, 'traject_yaw'].append(RelativeTrans['yaw'])  # degrees

            self._importance_weighting(Meas_Z_t)
        
    def _importance_weighting(self, Meas_Z_t):
        self.logger.info('Importance Weighting')
        self.Particle_DFrame['N_weight'] = self.Particle_DFrame['N_weight'] *(1- self.Particle_DFrame['meas_likly'])       
        # Weights  need to be normalized
        # Apply softmax to handle negative weights
        self.Particle_DFrame['N_weight'] = np.exp(self.Particle_DFrame['N_weight'])/ \
                                            np.sum(np.exp(self.Particle_DFrame['N_weight']))
        
        # particle with the highest weight to update Map
        self.best_particle = self.Particle_DFrame.meas_likly.idxmin()
        Est_X_t = self.Particle_DFrame.loc[self.best_particle].to_dict()       
        self.prev_scan_update = Est_X_t.copy()
        self._set_trajectory(Est_X_t)
        #self.OG.Update(Est_X_t, Meas_Z_t, True)

        # Check if resampling is required
        Den = self.Particle_DFrame['N_weight'] ** 2
        N_eff = 1 / Den.sum()
        if np.any(self.Particle_DFrame['N_weight']==np.nan):
            print(self.Particle_DFrame['N_weight'] )
        if N_eff < (self.Particle_Cnt / 2):
            self._random_resample()
        else:
            self.logger.info('Resampling Not required')

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
            try:       
                self._set_groundtruth(GPS_Z_t, IMU_Z_t, Meas_X_t)
                self.logger.info('RBPF_SLAM_Iteration.......%d for time = %f(s)', self.iteration, Meas_X_t['t'])
                
                if 'Range_XY_plane' not in Meas_Z_t.keys():
                    Meas_Z_t = Lidar_3D_Preprocessing(Meas_Z_t)
                if self.iteration == 0:
                    self.OG = Map()
                    self.SM = RTCSM(self.OG, Meas_X_t, Meas_Z_t)
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
                
            except Exception as e:
                print(e)
                raise(RuntimeWarning)
            finally:
                self.plot_results()                

    # Helpers
    def _set_groundtruth(self, GPS_Z_t, IMU_Z_t, Meas_X_t):
        self.True_x.append(GPS_Z_t['long'])
        self.True_y.append(GPS_Z_t['lat'])
        self.True_yaw.append(IMU_Z_t['yaw'])
        self.True_v.append( Meas_X_t['v'])
        self.True_acc.append( Meas_X_t['acc'])
        self.odom_x.append( Meas_X_t['x'])
        self.odom_y.append( Meas_X_t['y'])
        self.odom_yaw.append( Meas_X_t['yaw'])
        self.steer.append(np.rad2deg(Meas_X_t['steer']))
        self.time.append(Meas_X_t['t'])
        
    def _set_trajectory(self,Est_X_t):
        self.predict_x.append((self.Particle_DFrame['x'].to_numpy()))
        self.predict_y.append((self.Particle_DFrame['y'].to_numpy()))
        self.predict_yaw.append((self.Particle_DFrame['yaw'].to_numpy()))
        self.corrected_x.append(Est_X_t['x'])
        self.corrected_y.append(Est_X_t['y'])
        self.corrected_yaw.append(Est_X_t['yaw'])
        self.corrected_v.append(Est_X_t['v'])
        
    def plot_results(self):
        fig,axs = plt.subplots(2,3)
        #axs[0,0].plot(self.corrected_x, self.corrected_y, 'r+',label='corrected', markersize=1)
        axs[0,0].plot(self.True_x,self.True_y,'g.', markersize=1)
        axs[0,0].grid('on')
        axs[0,0].set_xlabel('Longitude')
        axs[0,0].set_ylabel('Latitude')
        axs[0,0].set_title("Ground Truth-XY-GPS")
        #axs[0,0].legend(loc='best')
        
        #prediction
        #axs[0,1].scatter(self.predict_x,self.predict_y,'r.' ,label='predicted', markersize=1)
        axs[0,1].plot(self.odom_x, self.odom_y, 'k*', label='Odom', markersize=1)
        #correction
        axs[0,1].plot(self.corrected_x, self.corrected_y, 'r+',label='corrected', markersize=1)
        axs[0,1].set_title("XY-Odom, MM,  SM")
        #axs[0,1].legend(loc='best')     
        axs[0,1].grid('on')

        # Orientation Comparison
        axs[1,0].plot(self.steer,'b.',label='steer', markersize=1)
        #axs[1,0].set_title('Yaw(degrees)')
        axs[1,0].grid('on')
        axs[1,0].set_ylabel('Yaw(degrees)')
        #axs[1,0].legend(loc='best')
        
        axs[1,1].plot(self.True_yaw,'g.',label='GT', markersize=1)        
        #axs[1,1].scatter(self.predict_yaw,'r.' ,label='predicted', markersize=1)
        axs[1,1].plot(self.odom_yaw, 'k*', label='Odom', markersize=1)
        axs[1,1].plot(self.corrected_yaw,'r+',label='corrected', markersize=1)
        axs[1,1].grid('on')
        #axs[1,1].legend(loc='best')
        
        #Vel
        axs[0,2].plot(self.True_v,'g.',label='GT', markersize=1)  
        axs[0,2].plot(self.corrected_v,'r+',label='corrected', markersize=1)
        
        #acc
        axs[1,2].plot(self.True_acc,'g.',label='GT', markersize=1)  
        h,l = axs[1,1].get_legend_handles_labels()
        fig.legend(h,l, loc='upper left')
        plt.tight_layout()
        plt.show()

    def _gps_2_XY(self):
        # In Meters
        lon2, lat2 = self.True_x[0], self.True_y[0]
        lon1, lat1 = self.True_x[-2], self.True_y[-2]
        dx = (lon1 - lon2) * 40000 * math.cos((lat1 + lat2) * math.pi / 360) / 360
        dy = (lat1 - lat2) * 40000 / 360
        return dx * 1000, dy * 1000
    
    def _particle_trajectory(self):
        plt.figure()
        for i in range(self.Particle_Cnt):
            plt.plot(self.Particle_DFrame.at[i,'traject_x'],self.Particle_DFrame.at[i,'traject_y'],label = i)
        plt.legend(loc = 'best')
        plt.show()
        
    def error_metrics(self):
        # corrected vs predicted
        # mean difference of location
        plt.figure()
        pos_diff_x = np.subtract(self.corrected_x, np.mean(self.predict_x))
        pos_diff_y = np.subtract(self.corrected_y, np.mean(self.predict_y))
        yaw_diff = np.subtract(self.corrected_yaw, np.mean(self.predict_yaw))
        plt.plot(pos_diff_x,legend = 'X_diff')
        plt.plot(pos_diff_y,legend = 'Y_diff')
        plt.plot(yaw_diff,legend = 'Yaw_diff')
        plt.legend(loc = 'best')
        print(f"The mean of diff b/t prediction and correction: X:{np.mean(pos_diff_x)}, Y:{np.mean(pos_diff_y)},Yaw:{np.mean(yaw_diff)}")

        # corrected vs odom
        plt.figure()
        pos_diff_x = np.subtract(self.corrected_x, self.odom_x)
        pos_diff_y = np.subtract(self.corrected_y, self.odom_y)
        yaw_diff = np.subtract(self.corrected_yaw, self.odom_yaw)
        plt.plot(pos_diff_x,legend = 'X_diff')
        plt.plot(pos_diff_y,legend = 'Y_diff')
        plt.plot(yaw_diff,legend = 'Yaw_diff')
        plt.legend(loc = 'best')
        print(f"The mean of diff b/t Odom and corrected: X:{np.mean(pos_diff_x)}, Y:{np.mean(pos_diff_y)},Yaw:{np.mean(yaw_diff)}")
        
        # 
    def groundtruth(self,Meas_X_t, Meas_Z_t, GPS_Z_t, IMU_Z_t):
        #odom
        self.True_x.append(Meas_X_t['x'])
        self.True_y.append(Meas_X_t['y'])
        self.True_yaw.append(Meas_X_t['yaw'])
        
        #GPS
        self.predict_x.append(GPS_Z_t['long'] )
        self.predict_y.append(GPS_Z_t['lat'] )
        self.predict_yaw.append(IMU_Z_t['yaw'])
        
        
if __name__ == '__main__':
    from main.ROSBag_decode import ROS_bag_run
    ROS_bag_run()