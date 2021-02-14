# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:02:41 2020

@author: MangalDeep
"""
import numpy as np
import logging

from utils.tools import Lidar_3D_Preprocessing
from .particle import Particle
class Gmapping():
    def __init__(self):
        self.iteration = 0
        self.particleList  = []

    def genrate_particles(self,Meas_X_t, N):
        self.Particle_cnt =N
        self.particleList = [Particle(Meas_X_t) for _ in range(N)]
        print('particle generated')

    def main(self, Meas_X_t, Meas_Z_t, IMU_Z_t):
        Meas_X_t['yaw_dot'] = IMU_Z_t['ang_vel']
        cmdIn = np.array([Meas_X_t['yaw_dot'], Meas_X_t['acc'], self.Meas_X_t_1['acc']]).reshape(3,)
        dt=Meas_X_t['t'] - self.Meas_X_t_1['t']
          
        for Particle in self.particleList:
            st_prime = Particle.motion_prediction(cmdIn, dt)

            GlobalTrans, RelativeTrans = Particle.scan_match(st_prime, Meas_Z_t, self.Meas_Z_t_1)

            if RelativeTrans['error'] > 5:
                #compute st, w
                #Update the exsiting Map
                pass
            else:
                # compute the Gaussian proposal

                # sample around the mode
                Gaus_sampl = st_prime[0:3]+ np.random.rand(3,10)
                lykly = []
                for j in range(Gaus_sampl.shape[1]):
                    GT,RT = Particle.scan_match(Gaus_sampl[:,j], Meas_Z_t, self.Meas_Z_t_1)
                   #self.Particle_DFrame['N_weight'] = np.exp(GT['error'])/ np.sum(np.exp(GT['error']))
                    lykly.append(1/GT['error'])

                for j in range(Gaus_sampl.shape[1]):
                    Particle.mu += Gaus_sampl[:,j] * lykly[j]
                    Particle.norm += lykly[j]
                
                Particle.mu = Particle.mu / Particle.norm

                for j in range(Gaus_sampl.shape[1]):
                    Particle.sigma += (Gaus_sampl[:,j] - Particle.mu) @ (Gaus_sampl[:,j] - Particle.mu).T

                Particle.sigma = Particle.sigma/Particle.norm

                #Sample Pose from Guas Approx
                Particle.st = Particle.mu + np.random.randn()*Particle.sigma

                #update importance weight
                Particle.w = Particle.w* Particle.norm
        
    def run(self,Meas_X_t, Meas_Z_t, GPS_Z_t, IMU_Z_t):
            # try:       
        if 'Range_XY_plane' not in Meas_Z_t.keys():
            Meas_Z_t = Lidar_3D_Preprocessing(Meas_Z_t)
        if self.iteration == 0:
            self.genrate_particles(Meas_X_t, 10)
            self.Meas_X_t_1 = Meas_X_t.copy()
            self.iteration += 1
            self.Meas_Z_t_1 = Meas_Z_t.copy()
            return None
        # Check Timeliness
        assert (self.Meas_X_t_1['t'] - Meas_X_t['t']) <= 1, "Time difference is very high"
        self.main(Meas_X_t, Meas_Z_t, IMU_Z_t)
        self.Meas_X_t_1 = Meas_X_t.copy()
        self.Meas_Z_t_1 = Meas_Z_t.copy()
        self.iteration += 1

if __name__ =='__main__':
    from main.ROSBag_decode import ROS_bag_run
    ROS_bag_run()
