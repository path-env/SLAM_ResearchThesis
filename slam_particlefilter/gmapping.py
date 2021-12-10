# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:02:41 2020

@author: MangalDeep
"""
import functools
from warnings import catch_warnings
import numpy as np
import logging
import concurrent

from libs.observation_models import Likelihood_Field_Observation_Model as lklyMdl
from utils.tools import Lidar_3D_Preprocessing
from slam_particlefilter.particle import Particle
from utils.tools import softmax, normalize
from libs.occupancy_grid import Map
from libs.scan_matching import ICP, RTCSM, Scan2Map
# from main.ROSBag_decode_gmap import ROS_bag_run
class Gmapping():
    def __init__(self, plotter, N):
        self.iteration = 0
        self.particleList  = []
        self.aly = plotter
        self.OG = Map()
        self.particleCnt = N
        self.parl = Parallel()
        self.SM_method = 0
        self.model_used = np.array([0,0,0])
        
    def noise_matrix(self, No_sample, var):
        # var = np.diag([0.1,0.1,0.1,0.1])
        noise = np.random.randn(No_sample,1)*0.1
        for _ in range(3):
            noise = np.hstack((noise, np.random.randn(No_sample,1)*0.1))
        noise[:,3] = 0
        # noise = noise *0.1 #@ var
        return noise

    def genrate_particles(self,Meas_X_t, Meas_Z_t, GPS_Z_t):
        self.OG.Update(Meas_X_t, Meas_Z_t,True)
        if self.SM_method == 2:
            self.SM = RTCSM(self.OG,Meas_X_t, Meas_Z_t)
        else:
            self.SM = ICP() #GO_ICP(Meas_Z_t, Meas_X_t)
        # self.SM = Scan2Map(self.OG)
        self.particleList = [Particle(Meas_X_t, Meas_Z_t, GPS_Z_t, self.OG, self.SM, cnt) for cnt in range(self.particleCnt)]

    def main(self, Meas_X_t, Meas_Z_t, IMU_Z_t, GPS_Z_t):
        cmdIn = np.array([IMU_Z_t['ang_vel'], Meas_X_t[-3], self.Meas_X_t_1[-3], self.IMU_Z_t_1['ang_vel']]).reshape(4,)
        dt=Meas_X_t[-1] - self.Meas_X_t_1[-1]
        dd = GPS_Z_t[:2]
        dd.append(Meas_X_t[2])
        dd.append(Meas_X_t[3])
        # self.particleList, self.model_used = self.parl._parallelize(self.particleList, cmdIn, Meas_Z_t,self.Meas_Z_t_1, dt, self.OG.MapIdx_G,
        #                                             self.prev_scan_update,dd)
        confi = []
        ###
        for P in self.particleList:
            # print(f"Particle {P.id} is being processed")
            st_prime = P.motion_prediction(cmdIn, dt)
            GT,GT_Lst = P.scan_match(st_prime,self.prev_scan_update, Meas_Z_t, self.Meas_Z_t_1, method =self.SM_method)
            print(GT['error'])
            confi.append(GT['error'])
            MapIdx_G = self.OG.MapIdx_G.copy()
            MapIdx_G[[1,0],:] = MapIdx_G[[0,1],:]
            if GT['error'] > P.err_thresh: #ICP SVD= 0.1. #ICP LS = 0.0015 
                #ScanMatch results very poor , use the motion model prediction
                P.st = st_prime
                P.w = P.w * lklyMdl(Meas_Z_t.to_numpy().T, P.st ,MapIdx_G)
                self.model_used[0]+=1
            else:
                # compute the Gaussian proposal
                st_hat = GT_Lst[:3]
                st_hat = np.append(st_hat, Meas_X_t[3:4])
                diff = st_hat - GPS_Z_t
                # print(f'diff1:{(diff)}')
                sample_cnt = 20
                Gaus_sampl = st_hat[0:4] + self.noise_matrix(sample_cnt, P.sigma)
                # reinitialize
                lykly = np.array([])
                P.mu = np.array([0.,0.,0.,0.])
                P.sigma = np.array([0.,0.,0.,0.])
                P.norm = 0.
                mu_mat = np.zeros((Gaus_sampl.shape[0], 4))
                sig_mat = np.zeros((Gaus_sampl.shape[0],0))
                cov_mat = np.zeros((Gaus_sampl.shape[0],4))
                #sub sample in the region found by scan matching
                #compute Mean
                for j in range(Gaus_sampl.shape[0]):
                    meas_lykly = lklyMdl(Meas_Z_t.to_numpy().T , Gaus_sampl[j],MapIdx_G)
                    # GT,_ = P.scan_match(Gaus_sampl[j], Meas_Z_t, self.Meas_Z_t_1)
                    mu_mat[j,:] = Gaus_sampl[j]
                    lykly = np.append(lykly,meas_lykly)
                # print(np.array(lykly))
                # lykly = np.ones((20,1))*0.05
                norm = sum(lykly)
                # normalize(np.ones(10))
                lykly = softmax(normalize(np.array(lykly)))
                mu_mat = mu_mat * lykly.reshape(Gaus_sampl.shape[0],1)
                P.norm = lykly.sum()
                P.mu = np.sum(mu_mat,axis =0)/P.norm
                 #compute Variance               
                for j in range(Gaus_sampl.shape[0]):
                    diff = Gaus_sampl[j] - P.mu
                    cov_mat[j,:] = (diff.T @ diff)*lykly[j]
                
                P.sigma = np.sum(cov_mat, axis=0)/P.norm
                # print(f'Type1:{P.sigma}')
                # P.sigma = np.cov(Gaus_sampl.T, aweights=lykly.flatten()) #covariance
                # print(f'Type2:{P.sigma}')
                #Sample Pose from Guas Approx
                P.st = P.mu #+ np.random.rand(4)*P.sigma
                diff = P.st[:3] - dd[:3] 
                # print(f'###diff2:{((diff))}###')
                # P.st = np.sum(P.st, axis=0)/4
                #update importance weight
                # P.st = st_hat
                P.w *= norm
                self.model_used[1]+=1
                # print(f'Partcile {P.id} done')
        ###
        part_w = np.array([])
        for P in self.particleList:
           part_w =np.append(part_w,P.w)
        # print(part_w)
        part_w = softmax(normalize(part_w))
        for i,P in enumerate(self.particleList):
            P.w = part_w[i]
        heavyPart = self.particleList[part_w.argmax()].st
        self._est_state(Meas_Z_t, Meas_X_t, heavyPart, dd)
        # Find N_eff       
        n_eff = 1/np.sum(part_w**2)
        #resample
        if n_eff < self.particleCnt/2:
            self._random_resample(part_w, Meas_Z_t)

    def _est_state(self, Meas_Z_t, Meas_X_t, heavyPart, dd):
        mu = np.array([0.,0.,0.,0.])
        var = np.zeros((1,4), dtype=np.float32)
        norm = 0.
        # weighed sum of each particle
        for P in self.particleList:
            mu += P.st*P.w
            var += P.sigma*P.w
            norm += P.w
        mu = mu/norm
        diff = mu[:3] - dd[:3] 
        print(f'diff:{(diff)}')
        self.prev_scan_update = mu
        self.aly._set_trajectory(mu,heavyPart)
        self.OG.Update(mu, Meas_Z_t,True)
        if self.SM_method == 2:
            self.SM.updateTargetMap(mu, Meas_Z_t)
        
    def _random_resample(self, part_w, Meas_Z_t):
        # Sample with replacement w.r.t weights
        self.model_used[2]+=1
        Choice_Indx = np.random.choice(np.arange(self.particleCnt), self.particleCnt, replace=True,
                                       p=part_w)
        self.particleList = np.array(self.particleList)
        self.particleList = self.particleList[Choice_Indx]
        NewParticleLst = []
        for i,P in enumerate(self.particleList):
            NP = Particle(P.st, Meas_Z_t,P.st, self.OG, self.SM, [P.id,i])
            NP.w = 1/self.particleCnt
            NewParticleLst.append(NP)
        self.particleList = NewParticleLst

    def run(self,Meas_X_t, Meas_Z_t, GPS_Z_t, IMU_Z_t, SM_method):
            # try:
        self.SM_method = SM_method    
        print(f"Time - {Meas_X_t['t']}sec, Iteration:{self.iteration}")   
        if 'Range_XY_plane' not in Meas_Z_t.keys():
            Meas_Z_t = Lidar_3D_Preprocessing(Meas_Z_t)
        Meas = np.array([Meas_X_t['x'], Meas_X_t['y'], Meas_X_t['yaw'], Meas_X_t['v'], Meas_X_t['acc'], Meas_X_t['steer'], Meas_X_t['t']])
        if self.iteration == 0:
            self.genrate_particles(Meas,Meas_Z_t, GPS_Z_t)
            self.Meas_X_t_1 = Meas.copy()
            self.IMU_Z_t_1 = IMU_Z_t.copy()
            self.Meas_Z_t_1 = Meas_Z_t.copy()
            self.prev_scan_update = self.Meas_X_t_1[:4]
            self.iteration += 1
            return None
        # Check Timeliness
        self.aly.set_groundtruth(GPS_Z_t, IMU_Z_t, Meas_X_t)
        assert (self.Meas_X_t_1[-1] - Meas[-1]) <= 1, "Time difference is very high"
        self.main(Meas, Meas_Z_t, IMU_Z_t, GPS_Z_t)
        self.Meas_X_t_1 = Meas.copy()
        self.IMU_Z_t_1 = IMU_Z_t.copy()
        self.Meas_Z_t_1 = Meas_Z_t.copy()
        self.iteration += 1
        self.aly.plot_results()

# if __name__ =='__main__':
#     from main.ROSBag_decode import ROS_bag_run
#     ROS_bag_run()

class Parallel:
    def __init__(self) -> None:
        self.model_used = np.array([0,0,0])

    def _parallelize(self,PartcileList,cmdIn, Meas_Z_t, Meas_Z_t_1,dt, MapIdx_G,prev_scan_update,dd):
        particles = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # results = [executor.submit(self._particleAnalysis, P,cmdIn, Meas_Z_t,Meas_Z_t_1, dt,MapIdx_G,prev_scan_update,dd) for P in PartcileList]
            results = executor.map(functools.partial(self._particleAnalysis,cmdIn, Meas_Z_t,Meas_Z_t_1, dt,MapIdx_G,prev_scan_update,dd),PartcileList)
            for p,i,j in results:
                particles.append(p)
                self.model_used[0]+=i
                self.model_used[1]+=j
        return particles, self.model_used
    
    def _particleAnalysis(self, cmdIn, Meas_Z_t,Meas_Z_t_1, dt,MapIdx_G,prev_scan_update,dd, P):
        MM,SM =0,0
        # print(f"Particle {P.id} is being processed")
        st_prime = P.motion_prediction(cmdIn, dt)
        GT,GT_Lst = P.scan_match(st_prime,prev_scan_update, Meas_Z_t, Meas_Z_t_1, method =2)
        # print(GT['error'])
        # confi.append(GT['error'])
        MapIdx_G = MapIdx_G.copy()
        MapIdx_G[[1,0],:] = MapIdx_G[[0,1],:]
        if GT['error'] > P.err_thresh: #ICP SVD= 0.1. #ICP LS = 0.0015 
            #ScanMatch results very poor , use the motion model prediction
            P.st = st_prime
            P.w = P.w * lklyMdl(Meas_Z_t.to_numpy().T, P.st ,MapIdx_G)
            MM+=1
        else:
            # compute the Gaussian proposal
            st_hat = GT_Lst[:3]
            st_hat = np.append(st_hat, dd[3:4])
            diff = st_hat - dd
            # print(f'diff1:{(diff)}')
            sample_cnt = 20
            Gaus_sampl = st_hat[0:4] + self.noise_matrix(sample_cnt, P.sigma)
            # reinitialize
            lykly = np.array([])
            P.mu = np.array([0.,0.,0.,0.])
            P.sigma = np.array([0.,0.,0.,0.])
            P.norm = 0.
            mu_mat = np.zeros((Gaus_sampl.shape[0], 4))
            sig_mat = np.zeros((Gaus_sampl.shape[0],0))
            cov_mat = np.zeros((Gaus_sampl.shape[0],4))
            #sub sample in the region found by scan matching
            #compute Mean
            for j in range(Gaus_sampl.shape[0]):
                meas_lykly = lklyMdl(Meas_Z_t.to_numpy().T , Gaus_sampl[j],MapIdx_G)
                # GT,_ = P.scan_match(Gaus_sampl[j], Meas_Z_t, self.Meas_Z_t_1)
                mu_mat[j,:] = Gaus_sampl[j]
                lykly = np.append(lykly,meas_lykly)
            # print(np.array(lykly))
            # lykly = np.ones((20,1))*0.05
            norm = sum(lykly)
            # normalize(np.ones(10))
            lykly = softmax(normalize(np.array(lykly)))
            mu_mat = mu_mat * lykly.reshape(Gaus_sampl.shape[0],1)
            P.norm = lykly.sum()
            P.mu = np.sum(mu_mat,axis =0)/P.norm
                #compute Variance               
            for j in range(Gaus_sampl.shape[0]):
                diff = Gaus_sampl[j] - P.mu
                cov_mat[j,:] = (diff.T @ diff)*lykly[j]
            
            P.sigma = np.sum(cov_mat, axis=0)/P.norm
            # print(f'Type1:{P.sigma}')
            # P.sigma = np.cov(Gaus_sampl.T, aweights=lykly.flatten()) #covariance
            # print(f'Type2:{P.sigma}')
            #Sample Pose from Guas Approx
            P.st = P.mu #+ np.random.rand(4)*P.sigma
            diff = P.st[:3] - dd[:3] 
            # print(f'###diff2:{((diff))}###')
            # P.st = np.sum(P.st, axis=0)/4
            #update importance weight
            # P.st = st_hat
            P.w *= norm
            SM+=1
            # print(f'Partcile {P.id} done')
        return P,MM,SM

    def noise_matrix(self, No_sample, var):
        # var = np.diag([0.1,0.1,0.1,0.1])
        noise = np.random.randn(No_sample,1)*0.1
        for _ in range(3):
            noise = np.hstack((noise, np.random.randn(No_sample,1)*0.1))
        noise[:,3] = 0
        # noise = noise *0.1 #@ var
        return noise