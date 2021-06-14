# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 09:59:08 2020

@author: MangalDeep
"""
import numpy as np
import sys
from numpy.core.fromnumeric import shape
from numpy.lib.stride_tricks import _maybe_view_as_subclass
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.linalg import norm
import scipy as sp
import matplotlib.pyplot as plt
import logging
from sklearn.neighbors import NearestNeighbors
from sksurgerygoicppython import GoICP, POINT3D, ROTNODE, TRANSNODE

from utils.tools import rotate , translate, normalize, softmax, poseComposition

def CoordTrans(Meas,X_t, X_t_1,TargetFrame='G'):
    if TargetFrame == 'G':
        X_t_1 =  {'x':0 , 'y':0 , 'yaw':0}
    #elif TargetFrame == 'R_':
    Meas_Trans = {}
    Z = Meas.T
    dx = X_t['x'] - X_t_1['x']
    dy = X_t['y'] - X_t_1['y']
    dyaw =np.deg2rad(X_t['yaw'] - X_t_1['yaw'])
    Meas_mat = rotate(dyaw) @ Z[:2,:] + translate(dx, dy)
    Meas_Trans['x'] = Meas_mat[0,:]
    Meas_Trans['y'] = Meas_mat[1,:]
    Meas_Trans['Azimuth'] = np.rad2deg(np.arctan2(Meas_Trans['y'] , Meas_Trans['x']))
    Meas_Trans['Range_XY_plane']= np.hypot(Meas_Trans['x'],Meas_Trans['y'])
    return Meas_Trans, Meas_mat
    
class Scan2Map():
    #Naming Convention of variables:
        #Z - Measurement
        #k - Time of measurement
        #R_/RR/G - R- / R+ /Global Coordinate Frame
    def __init__(self):
        self.KernelSize = 10
        self.logger = logging.getLogger('ROS_Decode.SM_ND')
        self.logger.info("ScanMatching Initialized")
               
    def CreateNDMap(self,OG,Pose):
        #Grab the map constructed so far and construct MAP ND
        self.Pose = Pose
        self.OG = OG
        self.Map_ND_k_1_G = self.Map_ND()
    
    def match(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1):
        self.logger.info("Matching the scan with current estimate to the Global Map")
        #Robot Coordinate Frame
        #ICP to derive R, T, phi
        R ,T_icp,phi_icp ,error =  self.ICP(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1)
        Meas = np.array([Meas_Z_t['x'] ,
                         Meas_Z_t['y']]).reshape(2,-1)
        
        Z_k_R_1 = rotate(phi_icp) @ Meas + T_icp
        Meas_R_1 = {'x': Z_k_R_1[1,:] ,'y': Z_k_R_1[2,:]}
        
        #In Global Frame
        phi = np.deg2rad(Est_X_t_1['yaw'])
        Z_k_G = rotate(phi) @ Z_k_R_1 + translate(Est_X_t_1['x'],Est_X_t_1['y'])
        
        #Robot pose in Global for time K     
        Est_X_k_G = rotate(phi) @ T_icp + translate(Est_X_t_1['x'],Est_X_t_1['y'])
        Est_phi_k_G = phi_icp + np.deg2rad(Est_X_t_1['yaw'])
       
        #Transform Scan to GCS
        Meas_Z_G = rotate(Est_phi_k_G) + translate(Est_X_k_G[0],Est_X_k_G[1]) 
        Scan_ND_k_G, ScanMap = self.Scan_ND(Meas_Z_G,Est_X_t) 
        
        #Find Correspondences
        Crs_Idx = self.Cell_Correspondence(self.Map_ND_k_1_G ,Scan_ND_k_G)
        
        #similiarity between two NDs using KL Divergence to find the matching Map
        Sim_score_MSE = self.Scan_Map_match(Scan_ND_k_G , self.Map_ND_k_1_G, Crs_Idx)
        
        if Sim_score_MSE > 5:
            # Assume where the bot starts is the global index (0,0,0)    
            #Iterative procedure to compute x,y,phi
            T, phi = self.Est_P_k_Gplus(Meas_Z_t,Est_X_t, Sim_score_MSE,Scan_ND_k_G, Crs_Idx)
        
        Est_X_k_Gplus = rotate(phi) @ T_icp + translate(Est_X_t['x'],Est_X_t['y'])
        Est_phi_k_Gplus = phi + np.deg2rad(Est_X_t['yaw'])
       
        return Sim_score_MSE, ScanMap , Est_X_k_Gplus, Est_phi_k_Gplus
    
    def Est_P_k_Gplus(self,Meas_Z_t,Est_X_t, MSE,Scan_ND_k_G, Crs_Idx):
        #Transform Scan to GCS
        #Meas_Z_G = CoordTrans(Meas_Z_t,Est_X_t,None,'G')
        #Scan_ND_k_G, ScanMap = self.Scan_ND(Meas_Z_G,Est_X_t)
        x,y,phi = 0,0,0
        ScanND_Gplus, MapND_Gplus = [{}],[{}]
        for jj in Crs_Idx:
            #Transform to GPlus using guessed
            #ScanND
            Meas = np.array([Scan_ND_k_G[jj]['x_mean'],
                             Scan_ND_k_G[jj]['y_mean']]).reshape(2,-1)
            Z_k_Gplus = rotate(phi) @ Meas + translate(x,y)      
            cov_k_Gplus = rotate(phi) @ Scan_ND_k_G[jj]['cov'] @ np.linalg.inv(rotate(phi))
            
            ScanND_Gplus[jj]['x_mean'], ScanND_Gplus[jj]['y_mean'] = Z_k_Gplus[0], Z_k_Gplus[1]
            ScanND_Gplus[jj]['cov'] = cov_k_Gplus
            
            #MapND
            Meas = np.array([self.Map_ND_k_1_G[jj]['x_mean'],
                             self.Map_ND_k_1_G[jj]['y_mean']]).reshape(2,-1)
            Z_k_Gplus = rotate(phi) @ Meas + translate(x,y)      
            cov_k_Gplus = rotate(phi) @ self.Map_ND_k_1_G[jj]['cov'] @ np.linalg.inv(rotate(phi))
            
            MapND_Gplus[jj]['x_mean'], MapND_Gplus[jj]['y_mean'] = Z_k_Gplus[0], Z_k_Gplus[1]
            MapND_Gplus[jj]['cov'] = cov_k_Gplus
        
        #Optimize for Maximum using Newton method            
        # def f(x):
        #     -o.5*(np.trace() )
        self.Scan_Map_match(ScanND_Gplus , MapND_Gplus, Crs_Idx)
        Similarity = np.sum(self.KL)
                
    def Compute_T_R(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1):
        #Transform Coordinates of measurements from  t to t-1
        (Meas_Z_R_,) = CoordTrans(Meas_Z_t, Est_X_t, Est_X_t_1, TargetFrame='R_')
        
        #Assume 1 to 1 correspondence
        X_t_mean = np.mean(Meas_Z_t_1['x'])
        Y_t_mean = np.mean(Meas_Z_t_1['y'])
        
        XMeas_Z_R__mean = np.mean(Meas_Z_R_['x'])
        YMeas_Z_R__mean = np.mean(Meas_Z_R_['y'])
        
        #MeanCentered
        X_t_MC = Meas_Z_t_1['x'] - X_t_mean
        Y_t_MC = Meas_Z_t_1['y'] - Y_t_mean
        
        XMeas_Z_R__MC = Meas_Z_R_['x'] - XMeas_Z_R__mean
        YMeas_Z_R__MC = Meas_Z_R_['y'] - YMeas_Z_R__mean
        
        #SVD Method
        q_t_1 = np.array([XMeas_Z_R__MC, YMeas_Z_R__MC]).reshape(2,-1)
        q_t = np.array([X_t_MC, Y_t_MC]).reshape(2,-1)
        H = np.matmul(q_t_1 ,q_t.T )
        u, s, vh = np.linalg.svd(H,full_matrices=True,compute_uv=True)
        
        #Find Roation
        R = np.matmul(u,vh)
        if np.linalg.det(R) <= -1:
            Warning('No Unique solution obtained')
        T = np.array([XMeas_Z_R__mean , YMeas_Z_R__mean]).reshape(2,1) - R@np.array([X_t_mean , Y_t_mean]).reshape(2,1)          
        error = np.sum(np.hypot(XMeas_Z_R__MC , YMeas_Z_R__MC)**2) + np.sum(np.hypot(X_t_MC , Y_t_MC)**2) - 2*np.sum(s)
        orientation = np.arctan2(R[1,0],R[0,0])
        return R,T,orientation,error
    
    def ICP(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 10,threshold = 0.01):
        self.logger.info("ICP........")
        assert Meas_Z_t['x'].shape == Meas_Z_t_1['x'].shape
        prev_error = 0
        lt = min(Meas_Z_t['x'].shape , Meas_Z_t_1['x'].shape  )
        t_1 = np.array([Meas_Z_t_1['x'], Meas_Z_t_1['x']]).reshape(-1,2)[:lt,:]
        t = np.array([Meas_Z_t['x'], Meas_Z_t['x']]).reshape(-1,2)[:lt,:]
        for i in range(Iter):
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(t)
            dist, idx = nbrs.kneighbors(t_1)
            
            dist = dist.ravel()
            idx = idx.ravel()
            idx = idx[dist<1]
            t['x'] = t[:,0]
            t_1['y'] = t_1[:,1]
            R,T,orientation,error = self.Compute_T_R(t,t_1 ,Est_X_t,Est_X_t_1)
            
            Meas_Z_t_1 = np.dot(R,Meas_Z_t) +T
            
            mean_dist_error = np.mean(dist)
            if np.abs(prev_error - mean_dist_error) < threshold:
                break
            prev_error = mean_dist_error
        R,T,orientation,error = self.Compute_T_R(t,t_1 ,Est_X_t,Est_X_t_1)
        return R,T,orientation,error
        
    def Map_to_GCS(self,Map,Pose_X_t = {'x':0 , 'y':0 , 'yaw':0}):
        #Only coordinates are processed not the entire matrix
        Map_G = [0,1]
        Global_CG = self.OG.getMapPivotPoint()        
        Map_G[0] =    Global_CG[0] - Map[0]
        Map_G[1] =    Global_CG[1] - Map[1]
        return Map_G[0] , Map_G[1]    
    
    def Scan_ND(self,Meas_Z_t,Pose_X_t):
        self.logger.info("Create ND Scan Map")
        ScanMap = (self.OG.getScanMap(Meas_Z_t,Pose_X_t))
        Scan_ND_Map,Scan_ND_k_G = self.getOccupancyInfo2(np.rot90(ScanMap))
        #self.OG.PlotMap(np.rot90(Scan_ND_Map,3),Pose_X_t,'Scan ND Map')
        return Scan_ND_k_G , ScanMap
               
    def getOccupancyInfo2(self,Map):
        OccVal = self.OG.l_occ
        KS = self.KernelSize
        #Appply on Map recursively creating cells j
        jj =0
        ND_cells =[]
        ND = np.zeros(Map.shape)
        for row in range(0,self.OG.Long_Length, KS):
            for col in range(0,self.OG.Lat_Width , KS):       
                Tmp = {}
                Tmp['j'] = jj
                map_extract = Map[row:KS+row,col:KS+col]
                ND_kernel= gaussian_filter(map_extract, sigma = 2) ## Only a representation 
                if np.any(map_extract >=OccVal*2):
                    ND[row:KS+row,col:KS+col] = map_extract
                    #ND -Mean
                    MC_coord = np.where(map_extract >= OccVal)
                    MC_coord = MC_coord + np.array([row,col]).reshape(2,-1)
                    MC_coord = self.Map_to_GCS(MC_coord)
                    
                    occ_coord = np.where(map_extract >= OccVal)
                    occ_coord = occ_coord + np.array([row,col]).reshape(2,-1)
                    occ_coord = self.Map_to_GCS(occ_coord)                    
                    #MC_coord = np.where(ND_kernel == np.mean(ND_kernel))
                    Tmp['cell_XIndx'] = (occ_coord[0])
                    Tmp['cell_YIndx'] = (occ_coord[1])
                    Tmp['NOD'] = len(np.where((map_extract)/OccVal > 0)[0] )
                    Tmp['x_mean'] = np.mean(MC_coord[0])
                    Tmp['y_mean'] = np.mean(MC_coord[1])
                    #ND Covariance
                    MC_x = Tmp['cell_XIndx'] - Tmp['x_mean']
                    MC_y = Tmp['cell_YIndx'] - Tmp['y_mean']
                    MC = np.array([MC_x , MC_y]).reshape(2,-1)
                    Tmp['cov'] = (np.matmul(MC, MC.T)) /max(Tmp['NOD'],1)  
                    ND_cells.append(Tmp)
                else:
                    ND_cells.append(Tmp)
                jj +=1
        return ND,ND_cells       
    
    def Map_ND(self):   
        self.logger.info("Create ND Global Map")        
        OccVal = self.OG.l_occ
        Map = np.rot90(self.OG.Local_Map)
        self.ND_map,Map_ND_k_1_G = self.getOccupancyInfo2(Map)
        #self.OG.PlotMap(np.rot90(self.ND_map,3),self.Pose,'Global ND Map')
        #self.PlotMapND(self.ND_map,'Global')
        return Map_ND_k_1_G

    def Cell_Correspondence(self, Map_NDs, Scan_NDs):
        MapIdx = np.where(*Map_NDs.keys() >1 )
        ScanIdx = np.where(*Scan_NDs.keys() >1 )
        return np.intersect1d(MapIdx, ScanIdx)
    
    def Scan_Map_match(self,Scan_ND , Map_ND):
        self.logger.info("Match Scan to Global Map")
        ND_Dim = 2
        Map_N_j = len(Map_ND)
        Scan_N_j = len(Scan_ND)
        assert Map_N_j == Scan_N_j
        # if Map_N_j != Scan_N_j:
        #     return False
        self.KL = []
        try:
            for jj in range(Scan_N_j):                
                if 'x_mean' in Map_ND[jj].keys() and 'x_mean' in Scan_ND[jj].keys():
                    if np.all(Map_ND[jj]['cell_XIndx'] == Map_ND[jj]['cell_XIndx'][0] ) or np.all(Map_ND[jj]['cell_YIndx'] == Map_ND[jj]['cell_YIndx'][0] ):
                        dist = 0
                        continue
                        Map_ND[jj]['cov'] = Map_ND[jj]['cov'] +0.1*np.random.randn(2,2)
                    
                    if np.all(Scan_ND[jj]['cell_XIndx'] == Scan_ND[jj]['cell_XIndx'][0] ) or np.all(Scan_ND[jj]['cell_YIndx'] == Scan_ND[jj]['cell_YIndx'][0] ):
                        dist = 0
                        continue
                        Scan_ND[jj]['cov'] = Scan_ND[jj]['cov'] +0.1*np.random.randn(2,2)
                        
                    Diff_x = Map_ND[jj]['x_mean']- Scan_ND[jj]['x_mean']
                    Diff_y = Map_ND[jj]['y_mean'] - Scan_ND[jj]['y_mean']     
                    #KL Divergence                
                    P2_2 = np.linalg.inv(Map_ND[jj]['cov']) 
                    P1 = np.trace( np.matmul(P2_2 ,Scan_ND[jj]['cov'] ))
                    P2_1 = np.array([Diff_x , Diff_y ])
                    P2_3 = P2_1.reshape(2,1)
                    num = np.linalg.det(Scan_ND[jj]['cov'])
                    den = np.linalg.det(Map_ND[jj]['cov'])             
                    P3 = np.log( np.abs(num)/ np.abs(den))
                    dist = -0.5* (P1 +  P2_1@P2_2@P2_3 - P3 -ND_Dim)[0]
                    Map_ND[jj]['S'] = dist
                elif 'x_mean' in Map_ND[jj].keys() and  'x_mean' not in Scan_ND[jj].keys():
                    continue
                elif 'x_mean' in Scan_ND[jj].keys() and 'x_mean' not in Map_ND[jj].keys():
                    continue
                else:   
                    continue 
                self.KL.append(dist)
            ##MSE
            #Error - Dist
            #Equation seems to do matrix square, hence skip
            # TO find only the mean value
            #   sss = [x for x in self.KL]     
            MSE = np.mean(self.KL)
            return MSE
        except Exception as e:
            self.logger.info('Exception',exc_info=True)
            self.logger.exception('%d,  %d',e,Map_ND,Scan_ND)
            return 0
        except:
            self.logger.info('error',exc_info=True)
            self.logger.error('%d,  %d',Map_ND,Scan_ND)
            return 0
        else:            
            return 0

    # def PlotMapND(self,Map,title):
    #     probMap = np.exp(Map)/(1.+np.exp(Map)) 
    #     plt.title(f"Gaussian Filtered Plot of {title} Map")
    #     plt.imshow(probMap, cmap='Greys')
    #     plt.draw()
    #     plt.show() 
        
        
    def EstimatePose(self):
        pass
    
    def CrctGCS(self):
        pass
    
    def CorrectPose(self):
        pass

class ICP():
    def _compute_T_R(self, Meas_Z_t_1, Meas_Z_t, weights):
        #Transform Coordinates of measurements from  t to t-1
        #(Meas_Z_R_,NP) = CoordTrans(Meas_Z_t, Est_X_t, Est_X_t_1, TargetFrame='R_')
        # weights  = Meas_Z_t - Meas_Z_t_1
        #Assume 1 to 1 correspondence
        #Old Pt
        X_t_1mean = np.mean(Meas_Z_t_1[0,:]) #np.mean(Meas_Z_t_1[0,:] * weights/np.sum(weights)) #
        Y_t_1mean = np.mean(Meas_Z_t_1[1,:]) #np.mean(Meas_Z_t_1[1,:] * weights/np.sum(weights)) #
        # Z_t_1mean = np.mean(Meas_Z_t_1[2,:])
        p_1mean = np.array([X_t_1mean,Y_t_1mean]).reshape(2,-1)
        #New pt
        X_t_mean = np.mean(Meas_Z_t[0,:]) #np.mean(Meas_Z_t[0,:] * weights/np.sum(weights)) #
        Y_t_mean = np.mean(Meas_Z_t[1,:]) #np.mean(Meas_Z_t[1,:] * weights/np.sum(weights)) #
        # Z_t_mean = np.mean(Meas_Z_t[2,:])
        p_mean = np.array([X_t_mean,Y_t_mean]).reshape(2,-1)
        
        #MeanCentered
        X_t_1MC = Meas_Z_t_1[0,:] - X_t_1mean
        Y_t_1MC = Meas_Z_t_1[1,:] - Y_t_1mean
        # Z_t_1MC = Meas_Z_t_1[2,:] - Z_t_1mean
        q_t_1 = np.array([X_t_1MC, Y_t_1MC ]).reshape(2,-1)
        
        X_tMC = Meas_Z_t[0,:] - X_t_mean
        Y_tMC = Meas_Z_t[1,:] - Y_t_mean
        # Z_tMC = Meas_Z_t[2,:] - Z_t_mean
        q_t = np.array([X_tMC, Y_tMC]).reshape(2,-1)*weights
        #SVD Method
        # q_t_1 = t_1MC.T
        # q_t = t_MC.T
        H = q_t_1 @ q_t.T
        u, s, vh = np.linalg.svd(H,full_matrices=True,compute_uv=True)
        
        #Find Roation
        R = u @ vh.T
        if np.linalg.det(R) <= -1:
            Warning('No Unique solution obtained')  
        T = p_mean - R@p_1mean 
        # error = np.sum(np.hypot(X_t_1MC , Y_t_1MC)**2 - 2*np.sum(s))
        # x_o = (R.T @ p_mean) - (R.T @ T )
        # error = np.sum(np.sqrt(np.square(q_t - R@(Meas_Z_t_1 - x_o))),  axis=1)
        error = np.sum(np.sqrt(np.square(q_t_1 - R@q_t)),axis=1)
        orientation = np.rad2deg(np.arctan2(R[0,1],R[1,1]))
        if np.any(np.isnan(T)):
            print('nan')
        return R,T,orientation,error, H, u, s, vh,p_1mean,p_mean
    
    def _closestPtCorrespondence(self,Meas_Z_t, Meas_Z_t_1, tolerance):
        # tolerance = np.hypot(Est_diff[0], Est_diff[1])
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Meas_Z_t.T)
        dist, p2_Idx = nbrs.kneighbors(Meas_Z_t_1.T)
        # if np.mean(dist)>5:
        #     print(np.mean(dist)
        # print(f"distance mean:{dist.mean()}, max:{dist.max()}")
        p1_Idx,_ = np.where(dist<= tolerance) # ICP LS = 0.5
        p2_Idx = p2_Idx[p1_Idx].flatten() #Use the closest point
        dist = dist[p1_Idx].flatten()
        weights = np.array([1])
        # weights = self.outlierRejection(indices, Colidx, dist)
        p2, p1 = Meas_Z_t[:,p2_Idx],Meas_Z_t_1[:,p1_Idx]
        if len(p2_Idx) == 0:
            print('No correspondense found')
        return p2,p1, weights

    def _matchRangePtCorrespondence(self,Meas_Z_t, Meas_Z_t_1, bound = 0.01):
        theta = np.rad2deg(np.arctan2(Meas_Z_t_1[1,:] , Meas_Z_t_1[0,:]))
        theta_hat = np.rad2deg(np.arctan2(Meas_Z_t[1,:] , Meas_Z_t[0,:]))
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(theta_hat.reshape(-1,1))
        dist, p2_Idx = nbrs.kneighbors(theta.reshape(-1,1))
        p1_Idx,_ = np.where(dist<= bound) 
        p2_Idx = p2_Idx[p1_Idx].flatten() #Use the closest point
        # for i,O in enumerate(theta):
        #     diff = np.abs(theta_hat-O)
        #     Idx = np.argmin(diff)
        #     if diff[Idx] <bound:
        #         p2_Idx.append(Idx)
        #         p1_Idx.append(i)
        p2, p1 = Meas_Z_t[:,p2_Idx], Meas_Z_t_1[:,p1_Idx]
        weights =1
        return p2, p1, weights

    def find_normal_vec(self,Meas_Z_t):
        delta_change = np.diff(Meas_Z_t, axis=0)
        norm_abs = np.linalg.norm(Meas_Z_t)
        unit_norm = delta_change/norm_abs
        dy, dx = delta_change[1], delta_change[0]

    def subsample(self,Meas_Z_t, Meas_Z_t_1):
        Meas_Z_t = np.round(Meas_Z_t,2)
        Meas_Z_t = np.unique(Meas_Z_t, axis=1)
        Meas_Z_t_1 = np.round(Meas_Z_t_1,2)
        Meas_Z_t_1 = np.unique(Meas_Z_t_1, axis=1)
        x1,x2 = Meas_Z_t_1.shape[1], Meas_Z_t.shape[1]
        P2, P1  = Meas_Z_t[:,0:min(x1,x2)],Meas_Z_t_1[:,0:min(x1,x2)]
        return P2, P1

    def outlierRejection(self,indices, Colidx, dist):
        weights = softmax(normalize(1/dist).flatten())
        idx = indices[np.where(dist>=1)]
        weights[idx] = 0
        # weights = np.array([1])
        return weights

    def match_SVD(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 400,threshold = 0.000001):
        Meas_Z_t = Meas_Z_t[[0,1],:]
        Meas_Z_t_1 = Meas_Z_t_1[[0,1],:]
        # if Meas_Z_t.shape != Meas_Z_t_1.shape:
        #     lt = min(Meas_Z_t_1.shape[1], Meas_Z_t.shape[1] )
        #     Meas_Z_t_1 = Meas_Z_t_1[:,0:lt]
        #     Meas_Z_t = Meas_Z_t[:,0:lt]      
        prev_error = 0
        alignmenterr = 10000
        loop_cnt = 0
        # Transform to the estimated pose
        # Meas_Z_t = rotate(Est_X_t[2]) @ Meas_Z_t + Est_X_t[0:2].reshape(2,-1)
        # Meas_Z_t_1 = rotate(Est_X_t_1[2]) @ Meas_Z_t_1 + Est_X_t_1[0:2].reshape(2,-1)
        actMeas = Meas_Z_t_1.copy()
        #Transform to the estimated relative pose
        # diff= Est_X_t_1 - Est_X_t
        # Meas_Z_t_1 = rotate(diff[2]) @Meas_Z_t_1 + diff[0:2].reshape(2,-1) 
        try:
            while np.any(np.abs(prev_error - alignmenterr) > threshold):
                prev_error = alignmenterr
                # Meas_Z_t, Meas_Z_t_1 = self.subsample(Meas_Z_t, Meas_Z_t_1)
                p2,p1, weights =self._closestPtCorrespondence(Meas_Z_t, Meas_Z_t_1, Est_X_t)
                R,T,orientation,alignmenterr,_,_,_,_,p_1mean,p_mean = self._compute_T_R(p2,p1, weights)                
                Meas_Z_t_1 = (R @ (Meas_Z_t_1[:,:]- p_mean) )+p_1mean  
                loop_cnt +=1
                # print(alignmenterr, orientation)
                if np.all(abs(alignmenterr)< threshold) or loop_cnt > Iter:
                    break   
            # if alignmenterr > 5: #Alignment impossible
            #     #print(f"Error beyond threshold @ {alignmenterr}")
            #     alignmenterr = None
            RelativeTrans = {'r':R,'T':T , 'yaw':orientation,'error':alignmenterr}
            R,T,orientation,error,_,_,_,_,_,_ = self._compute_T_R(Meas_Z_t_1,actMeas, weights)
            GlobalTrans_Lst = np.append(T,orientation)
            Trans_Lst = poseComposition(Est_X_t_1, GlobalTrans_Lst)
            GlobalTrans = {'r':rotate(Trans_Lst[2]),'T':Trans_Lst[:2].reshape(2,1) , 'yaw':Trans_Lst[2],'error':np.linalg.norm(error)}
            return GlobalTrans,Trans_Lst
        except Exception as e:
            print(e)

    def match_LS(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 40,threshold = 0.00001):
        # Using Gausss Newton approximation
        chi2_lst, X_lst = [], []
        bound = 1
        del_x = np.array([0.,0.,0.]).reshape(3,-1)
        x = (Est_X_t[:3] - Est_X_t_1[:3]).reshape(3,-1)
        x[2] = np.deg2rad(x[2])
        init_est = x.copy()
        # x = np.array([0.,0.,0.]).reshape(3,-1)
        p2, p1 =(Meas_Z_t[0:2,:], Meas_Z_t_1[0:2,:])
        # p2 = rotate(Est_X_t[2])@p2 + Est_X_t[0:2].reshape(2,-1)
        # p1 = rotate(Est_X_t_1[2])@p1 + Est_X_t_1[0:2].reshape(2,-1)
        P2, P1 = self.subsample(p2, p1)
        for t in range(Iter):
            chi2,deg_x = 0, np.rad2deg(x[2]            )
            # Apply transformation on pointcloud
            P1 = rotate(deg_x)@P1 + x[0:2]
            bound = max(0.1,bound*np.exp(-0.1*t))
            p2, p1, weights =self._closestPtCorrespondence(P2, P1 ,bound)
            # p2, p1, weights =self._matchRangePtCorrespondence(p2, p1,bound)
            if p2.shape[1]==0:
                Trans = {'error':np.inf}
                Trans_Lst = x.flatten()
                return Trans, Trans_Lst

            J = np.zeros((2,3,p1.shape[1]))
            H = np.zeros((3,3))
            b = np.zeros((3,1))
            e = p1[:2,:] - p2[:2,:]
            J[0,0,:] , J[1,1,:] =1,1
            J[0,2,:] = -np.sin(deg_x)*p1[0,:] - np.cos(deg_x)*p1[1,:] 
            J[1,2,:] = np.cos(deg_x)*p1[0,:]  + np.sin(deg_x)*p1[1,:] 
            for j in range(p1.shape[1]):
                H += J[:,:,j].T @ J[:,:,j]
                b += J[:,:,j].T @ e[:,j:j+1]
                chi2 += e[:,j:j+1].T @e[:,j:j+1]
            # print(f"The chi^2 error:{chi2}, matshape:{len(indices)}")
            chi2 = chi2/p1.shape[1]
            if np.abs(chi2) < threshold:
                x[2] = np.rad2deg(x[2])
                Trans_Lst = poseComposition(Est_X_t_1, x.flatten())      
                Trans = {'r':rotate(x[2]),'T': x[0:2], 'yaw':x[2],'error':chi2}
                return Trans, Trans_Lst
            del_x = -np.linalg.inv(H)@b
            x += del_x
            # print(x, chi2)
            # x[2] = (np.arctan2(np.sin(x[2]), np.cos(x[2])))
            chi2_lst.append(chi2.flatten())
            X_lst.append(x.copy())
        X_lst = np.array(X_lst)
        minIndx,_ = np.where(chi2_lst==min(chi2_lst))
        PoseDiff = X_lst[minIndx+1][0].flatten()
        PoseDiff[2] = np.rad2deg(PoseDiff[2])
        Trans_Lst = poseComposition(Est_X_t_1, PoseDiff)      
        error = max(min(chi2_lst),0.00001)
        Trans = {'r':rotate(Trans_Lst[2]),'T': Trans_Lst[:2].reshape(2,1), 'yaw':Trans_Lst[2],'error':error}
        return Trans,Trans_Lst
    
    def match_IDC(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 40,threshold = 0.001):
        self._matchRangePtCorrespondence(Meas_Z_t, Meas_Z_t_1)
        pass

    def match_p2plane(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 40,threshold = 0.000001): 
        # Find normals to the point on the point cloud.
        # --Draw a tangent to the curve measure the gradient of the tangent(y-y1) = m(x-x1)
        # --Normal to the curve is the tangent to the curve @ that point
        pass
class RTCSM():
    def __init__(self, og,Est_X_t, Meas_Z_t):
        self.OG = og
        self.dim = self.OG.max_lidar_r + self.OG.Roffset
        self.searchWindow_t = 4.0 # meters
        self.searchWindow_yaw = 90 #degrees
        self.highResMap_t_1,centre_pos = self._highResMap(Est_X_t, Meas_Z_t)
        #Est_X_t, Meas_Z_t = self._modifyDtype(Est_X_t, Meas_Z_t)      
        self.lowResMap_t_1 = self._lowResMap(self.highResMap_t_1, centre_pos, Est_X_t)
        self.updateTargetMap(Est_X_t, Meas_Z_t)
        self.Pos_var = 0.1
        self.ori_var = 0.1
        
    def match(self, Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1):
        #Construct High resolution map
        HMap,centre_pos = self._highResMap(Est_X_t, Meas_Z_t)
        #Est_X_t, Meas_Z_t = self._modifyDtype(Est_X_t, Meas_Z_t)
        
        #Construct Low resolution Map
        LMap = self._lowResMap(HMap, centre_pos, Est_X_t)
        Updt_X_t, confidence_ = self._search3DMatch( Est_X_t, Est_X_t_1, Meas_Z_t, centre_pos, LMap, 0.5,mode='L')
        Updt_X_t, confidence = self._search3DMatch(Updt_X_t, Est_X_t_1, Meas_Z_t, centre_pos, HMap, 0.05, mode= 'H')
        Trans = {'error':confidence, 'T':np.array([Updt_X_t[0], Updt_X_t[1]]).reshape(2,-1), 'yaw':Updt_X_t[2]}
        return Trans
    
    def _highResMap(self, Est_X_t, Meas_Z_t):        
        HMap,centre_pos,_,_ = self.OG.getScanMap(Meas_Z_t, Est_X_t)
        Xrange = np.arange(centre_pos[0][0]-self.dim ,  centre_pos[0][0]+self.dim)
        Yrange = np.arange(centre_pos[1][0]-self.dim ,  centre_pos[1][0]+self.dim)     
        
        Map_confined = HMap[Yrange[0]:Yrange[-1]+1, Xrange[0]:Xrange[-1]+1]
        # dist_to_grid = dist_to_grid[Xrange[0]:Xrange[-1], Yrange[0]:Yrange[-1]]
        #self.OG.PlotMap(np.rot90(Map_confined,3),Est_X_t,'High resolution Map')
        return Map_confined, centre_pos

    def _lowResMap(self,HMap, centre_pos, Est_X_t):
        LowResol = 3*self.OG.Grid_resol
        Map_X_len, Map_Y_Len = HMap.shape
        LowResolMap = np.zeros(HMap.shape)
        for row in range(0,Map_X_len,LowResol):
            for col in range(0,Map_Y_Len,LowResol):
                LowResolMap[row:row+LowResol, col:col+LowResol] = np.max(HMap[row:row+LowResol, col:col+LowResol])
        
        #self.OG.PlotMap(np.rot90(LowResolMap,3),Est_X_t,'Low resolution Map 3:1')        
        return LowResolMap
    
    def _search3DMatch(self, Est_X_t, Est_X_t_1, Meas_Z_t, centre_pos, Map, searchStep,mode= 'L'):                
        MeasMap = Map
        # create a search space
        if mode == 'L':
            Numcell = 5/searchStep #5
        else:
            Numcell = 0.2/searchStep #4
            #Numcell = 15
        Xrange = np.arange(-Numcell ,  Numcell+searchStep)#, searchStep)
        Yrange = np.arange(-Numcell ,  Numcell+searchStep)#, searchStep)
        x , y = np.meshgrid(Xrange,Yrange)
        if mode == 'H':
            TargetMap = self.HTgtMap
            ori_space = np.arange(-1,1,searchStep)
            Pos_search_mask = np.zeros((x.shape[0], x.shape[1]))
            Ori_search_mask = np.zeros((x.shape[0], x.shape[1]))
        else:
            TargetMap = self.HTgtMap
            ori_space = np.arange(-5,5,searchStep)
        
            EstDistDrvn = np.sqrt((Est_X_t_1[0] - Est_X_t[0])**2 + (Est_X_t_1[1] - Est_X_t[1])**2 )
            sq = np.sqrt((x*searchStep)**2 + (y*searchStep)**2)#, dtype=np.float32)
            rrv = np.abs(sq + EstDistDrvn)
            # sq[np.where(sq <EstDistDrvn)] = np.ceil(EstDistDrvn)
            Pos_search_mask = (-(1 / (2 * self.Pos_var**2)) * (sq + EstDistDrvn)**2)
            # Pos_search_mask = normalize(Pos_search_mask)
            Pos_search_mask[rrv>1] = -100
            #Pos_search_mask[Pos_search_mask < -25000] = -10000000
            if np.abs(Est_X_t[2]  - Est_X_t_1[2]) > 1:
                distv = np.sqrt(x**2 + y**2)
                distv[distv == 0] = 0.0001
                yaw = np.deg2rad(Est_X_t[2]  - Est_X_t_1[2])
                Ori_search_mask = np.arccos((x* np.cos(yaw) + y* np.sin(yaw)) / distv) 
                Ori_search_mask = -1 / (2 * self.ori_var ** 2) * np.square(Ori_search_mask)
            else:
                Ori_search_mask = np.zeros((x.shape[0], x.shape[1]))
            # Pos_search_mask = np.zeros((Map.shape[0], Map.shape[1]))
            # Ori_search_mask = np.zeros((Map.shape[0], Map.shape[1]))
        Mask = Pos_search_mask + Ori_search_mask
        #Mask = normalize(Mask)
        #Mask[Mask<-0.6] = 0
        x = x.reshape((x.shape[0], x.shape[1], 1))
        y = y.reshape((y.shape[0], y.shape[1], 1))
        Corr_Cost_Fn = np.zeros((len(ori_space), x.shape[0],x.shape[1]))
        Corr_cost = np.zeros(len(ori_space),)
        theta = np.zeros(len(ori_space),)
        Meas = {}
        #Meas['x'], Meas['y']= Meas_Z_t['x'], Meas_Z_t['y']
        plt.figure()
        for i,ori in enumerate(ori_space):
            Est= np.array([Est_X_t[0],Est_X_t[1] ,Est_X_t[2]+ori]).reshape(-1,1)
            #Meas = self._rotate(Est, Meas, searchStep)
            Temp_map = sp.ndimage.rotate(MeasMap, ori , reshape=False)
            Temp_map[Temp_map<0.3] = 0 
            # m = self.rotate(Est,Meas)
            Meas['x'], Meas['y']= np.where(Temp_map>0)
            uniqueRotatedPxPyIdx = np.unique(np.column_stack((Meas['x'], Meas['y'])), axis=0)
            rotatedPxIdx, rotatedPyIdx = uniqueRotatedPxPyIdx[:, 0], uniqueRotatedPxPyIdx[:, 1] 
            rotatedPxIdx = rotatedPxIdx.reshape(1, 1, -1) #- self.OG.max_lidar_r
            rotatedPyIdx = rotatedPyIdx.reshape(1, 1, -1) #-  self.OG.max_lidar_r
            rotatedPxIdx = (rotatedPxIdx + x +Est[0]).astype(int)
            rotatedPyIdx = (rotatedPyIdx + y +Est[1] +self.dim).astype(int)
            convResult = -TargetMap[rotatedPxIdx, rotatedPyIdx]

            convResultSum = np.sum(convResult, axis=2)
            # convResultSum = normalize(convResultSum)
            UncrtnMap = convResultSum + Mask
            theta[i] = Est[2]
            Corr_cost[i] = norm(np.abs(UncrtnMap))
            Corr_Cost_Fn[i,:,:] = UncrtnMap
            #Temp_map= sp.ndimage.rotate(MeasMap, ori , reshape=False)
            #Temp_map[Temp_map<0] = 0
            # self.OG.PlotMap(Temp_map,Est,'Temp_map')     
            # self.OG.PlotMap(np.rot90(Temp_map,0),Est,'tmp map')
            # plt.contour(Pos_search_mask)
            # plt.contour(Ori_search_mask)

            plt.contourf(UncrtnMap)
            plt.pause(0.001)
            plt.cla()

        # Find the best voxel in the Low resolution Map
        minIdx = np.unravel_index(Corr_Cost_Fn.argmin(), Corr_Cost_Fn.shape)
        confidence = np.sum(np.exp(Corr_Cost_Fn))
        dx, dy, dtheta = Xrange[minIdx[1]]*searchStep , Yrange[minIdx[2]]*searchStep, ori_space[minIdx[0]]
        print(dx, dy, dtheta, confidence)
        Updt_X_t = [Est_X_t[0] + dx, Est_X_t[1] + dy, Est_X_t[2] + dtheta]
        return Updt_X_t , confidence
        
    def covertMeasureToXY(self, Est_X_t, rMeasure):
        rads = np.linspace(Est_X_t[2] - self.OG.FOV / 2, Est_X_t[2]  + self.OG.FOV / 2,num=10)
        range_idx = rMeasure < self.OG.max_lidar_r
        rMeasureInRange = rMeasure[range_idx]
        rads = rads[range_idx]
        px = Est_X_t[0]  + np.cos(rads) * rMeasureInRange
        py = Est_X_t[1]  + np.sin(rads) * rMeasureInRange
        return px, py

    def rotate(self, Est_X, Meas):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        ox, oy = Est_X[0], Est_X[1]
        px, py = Meas['x'], Meas['y']
        qx = ox + np.cos(np.deg2rad(Est_X[2])) * (px - ox) - np.sin(np.deg2rad(Est_X[2])) * (py - oy)
        qy = oy + np.sin(np.deg2rad(Est_X[2])) * (px - ox) + np.cos(np.deg2rad(Est_X[2])) * (py - oy)
        xIdx = (((qx - (Est_X[0] - self.OG.max_lidar_r)) / 0.5)).astype(int)
        yIdx = (((qy - (Est_X[1] - self.OG.max_lidar_r)) / 0.5)).astype(int)
        return qx, qy

    def convertXYToSearchSpaceIdx(self, px, py, beginX, beginY, unitLength):
        xIdx = (((px - beginX) / unitLength)).astype(int)
        yIdx = (((py - beginY) / unitLength)).astype(int)
        return xIdx, yIdx

    def _rotate(self, Est_X, Meas, searchStep):
        RotatedMeas ={}
        Meas = np.array([Meas['x'], Meas['y']]).reshape(2,-1) #+ self.OG.max_lidar_r
        Meas = Est_X[:2] + rotate(Est_X[2])@ Meas# - Est_X[:2])
        RotatedMeas['x'] = ((Meas[0,:]) )# - Est_X[0] + self.OG.max_lidar_r)/searchStep).astype(int)
        RotatedMeas['y'] = ((Meas[1,:]) )# - Est_X[1] + self.OG.max_lidar_r)/searchStep).astype(int)
        ### bedug ####
        # www = np.zeros((500,500))
        # Meas = Meas.astype(int)
        # www[Meas[0,:], Meas[1,:]] = 1
        # plt.contourf(www)
        # plt.pause(0.001)
        # Est = np.array(Est_X) - self.OG.max_lidar_r
        # Idx = RotatedMeas - Est
        # RotatedMeas['Azimuth'] = np.rad2deg(np.arctan2(RotatedMeas['y'] , RotatedMeas['x']))
        # RotatedMeas['Range_XY_plane']= np.hypot(RotatedMeas['x'],RotatedMeas['y'])
        return RotatedMeas
    
    def updateTargetMap(self, Est_X_t, Meas_Z_t):
        self.Est_X_t_1 = Est_X_t
        self.Meas_Z_t_1 = Meas_Z_t  
        ExtractMap,_= self.OG.getExtractMap(Est_X_t)
        self.HTgtMap =  gaussian_filter(self.OG.LO_t_i, sigma =1)
        self.LTgtMap =  gaussian_filter(ExtractMap, sigma =1)
        
    def _modifyDtype(self, Est_X_t, Meas_Z_t):
        Meas_Z_t = np.array([Meas_Z_t['x'], Meas_Z_t['y']]).reshape(2,-1)
        Est_X_t = np.array([Est_X_t['x'], Est_X_t['y'], np.deg2rad(Est_X_t[2])]).reshape(3,1)
        return Est_X_t, Meas_Z_t
    
class GO_ICP():
    def __init__(self,Meas_Z_t, Est_X_t) -> None:
        self.go_icp = GoICP()
        X = np.array([Est_X_t['x'], Est_X_t['y'], Est_X_t['yaw']])
        self.Nm, self.PCL_t_1 = self._loadPCL(Meas_Z_t.to_numpy().T, X)
        
    def _loadPCL(self, Meas_Z_t, Est_X_t):
        Meas_Z_t = rotate(Est_X_t[2]) @ Meas_Z_t[:2,:] + Est_X_t[0:2].reshape(2,-1)
        Meas_Z_t = np.vstack((Meas_Z_t, np.zeros((1,Meas_Z_t.shape[1]))))
        Meas_list = Meas_Z_t.T.tolist()
        plist = []
        for x,y,z in Meas_list:
            pt = POINT3D(x,y,z)
            plist.append(pt)
        return len(Meas_list), plist

    def match(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 10,threshold = 0.0001):
        Nd, PCL_t = self._loadPCL(Meas_Z_t, Est_X_t)
        self.go_icp.loadModelAndData(self.Nm, self.PCL_t_1, Nd, PCL_t)
        self.go_icp.setDTSizeAndFactor(300, 2.0)
        self.go_icp.BuildDT()
        self.go_icp.Register()
        self.PCL_t_1 = PCL_t
        self.Nm = Nd
        GT = {'T': self.go_icp.optimalTranslation(), 'R':self.go_icp.optimalRotation()}
        RT = GT.copy()
        return GT, RT
        print(self.go_icp.optimalRotation())
        print(self.go_icp.optimalTranslation())

if __name__ == '__main__':
    R = 2 # in degress
    T = np.array([0.5,0]).reshape(2,1)
    Meas_Z_t_1 =np.array([[1,-1],[1,1]],dtype = np.float64)
    Meas_Z_t = np.array([[1,-1],[0,2]],dtype = np.float64)
    Meas_Z_t_1 = np.array([[2,3,-2,-3,-2,-3,2,3],[2,3,2,3,-2,-3,-2,-3]])
    Meas_Z_t = (rotate(R) @ Meas_Z_t_1 ) + T
    Est_X_t = np.array([1,0.,5],dtype = np.float64)
    Est_X_t_1 =np.array([0,0,0],dtype = np.float64)
    SM = ICP()   
    R,T,orientation,error = SM.match_LS(Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1)