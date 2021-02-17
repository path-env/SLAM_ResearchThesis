# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:02:41 2020

@author: MangalDeep
"""
from random import sample
import numpy as np
import logging

from utils.tools import Lidar_3D_Preprocessing
from slam_particlefilter.particle import Particle
from utils.tools import softmax, normalize
class Gmapping():
    def __init__(self, plotter):
        self.iteration = 0
        self.particleList  = []
        self.aly = plotter

    def noise_matrix(self, No_sample, var):
        noise = np.random.randn(4)
        for _ in range(No_sample-1):
            noise = np.vstack((noise, np.random.randn(4)))
        noise = noise * var
        return noise

    def genrate_particles(self,Meas_X_t, N):
        self.particleCnt =N
        self.particleList = [Particle(Meas_X_t) for _ in range(N)]

    def main(self, Meas_X_t, Meas_Z_t, IMU_Z_t):
        Meas_X_t['yaw_dot'] = IMU_Z_t['ang_vel']
        cmdIn = np.array([Meas_X_t['yaw_dot'], Meas_X_t['acc'], self.Meas_X_t_1['acc']]).reshape(3,)
        dt=Meas_X_t['t'] - self.Meas_X_t_1['t']

        part_w = []  
        for P in self.particleList:
            P.id = len(part_w)
            st_prime = P.motion_prediction(cmdIn, dt)
            GT, RT = P.scan_match(st_prime, Meas_Z_t, self.Meas_Z_t_1)

            if RT['error'] > 5:
                #compute st, w
                P.st = st_prime
                P.w = P.w * 1/RT['error']
            else:
                # compute the Gaussian proposal
                x_k = st_prime -  np.vstack((GT['T'], GT['yaw'], 0))
                # sample around the mode
                sample_cnt = 10
                Gaus_sampl = st_prime[0:4]+ self.noise_matrix(sample_cnt, P.sigma)
                # reinitialize
                lykly = []
                P.mu = np.array([0.,0.,0.,0.])
                P.sigma = np.array([0.,0.,0.,0.])
                P.norm = 0.

                for j in range(Gaus_sampl.shape[0]):
                    GT,_ = P.scan_match(Gaus_sampl[j], Meas_Z_t, self.Meas_Z_t_1)
                   #self.Particle_DFrame['N_weight'] = np.exp(GT['error'])/ np.sum(np.exp(GT['error']))
                    lykly.append((1-GT['error']))

                #lykly = np.array(lykly) - max(lykly)
                lykly = softmax(lykly)

                # Compute Mean
                P.norm = lykly.sum()
                temp = Gaus_sampl.T * lykly
                P.mu = temp.T
                P.mu = np.sum(P.mu, axis =0) / P.norm

                # Compute Variance
                # for j in range(Gaus_sampl.shape[1]):
                #     diff = (Gaus_sampl[j] - P.mu)
                #     P.sigma += ( diff @ (diff.reshape(4,1).T))*lykly[j]
                #P.sigma = P.sigma/P.norm
                diff = Gaus_sampl - P.mu
                temp = diff @ diff.T
                P.sigma = temp * lykly
                P.sigma = P.sigma /P.norm
                P.sigma = np.cov(Gaus_sampl.T, aweights=lykly)
                #Sample Pose from Guas Approx
                P.st = P.mu + np.random.randn()*P.sigma

                #update importance weight
                P.w = P.w * P.norm
            part_w.append(P.w)
                #Update Map for particle
                #Update sample set

        # find the particle with the max weight
        max_id = np.argmax(part_w)
        P = self.particleList[max_id]
        self.aly._set_trajectory(P.st)

        # Find N_eff
        part_w = softmax(part_w)
        part_w = np.array(part_w)
        n_eff = 1/np.sum(part_w**2)

        if n_eff < self.particleCnt/2:
            #resample
            self._random_resample(part_w)
    
    def _random_resample(self, part_w):
        # Sample with replacement w.r.t weights
        Choice_Indx = np.random.choice(np.arange(self.particleCnt), self.particleCnt, replace=True,
                                       p=part_w)
        self.particleList = self.particleList[Choice_Indx]
        self.P_DFrame['N_weight'] = 1 / self.particleCnt

    def run(self,Meas_X_t, Meas_Z_t, GPS_Z_t, IMU_Z_t):
            # try:       
        if 'Range_XY_plane' not in Meas_Z_t.keys():
            Meas_Z_t = Lidar_3D_Preprocessing(Meas_Z_t)
        if self.iteration == 0:
            self.genrate_particles(Meas_X_t, 10)
            self.Meas_X_t_1 = Meas_X_t.copy()
            self.Meas_Z_t_1 = Meas_Z_t.copy()
            self.iteration += 1
            return None
        # Check Timeliness
        assert (self.Meas_X_t_1['t'] - Meas_X_t['t']) <= 1, "Time difference is very high"
        self.main(Meas_X_t, Meas_Z_t, IMU_Z_t)
        self.Meas_X_t_1 = Meas_X_t.copy()
        self.Meas_Z_t_1 = Meas_Z_t.copy()
        self.iteration += 1
        self.aly.plot_results()

# if __name__ =='__main__':
#     from main.ROSBag_decode import ROS_bag_run
#     ROS_bag_run()
