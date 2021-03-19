# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 09:59:08 2020

@author: MangalDeep
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.linalg import norm
import scipy as sp
import matplotlib.pyplot as plt
import logging
from sklearn.neighbors import NearestNeighbors
from sksurgerygoicppython import GoICP, POINT3D, ROTNODE, TRANSNODE

from utils.tools import rotate , translate

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
    def _compute_T_R(self, Meas_Z_t_1, Meas_Z_t):
        #Transform Coordinates of measurements from  t to t-1
        #(Meas_Z_R_,NP) = CoordTrans(Meas_Z_t, Est_X_t, Est_X_t_1, TargetFrame='R_')
        
        #Assume 1 to 1 correspondence
        #New Pt
        X_t_1mean = np.mean(Meas_Z_t_1[0,:])
        Y_t_1mean = np.mean(Meas_Z_t_1[1,:])
        t_1mean = np.array([X_t_1mean,Y_t_1mean ]).reshape(2,-1)
        #Old pt
        X_t_mean = np.mean(Meas_Z_t[0,:])
        Y_t_mean = np.mean(Meas_Z_t[1,:])
        t_mean = np.array([X_t_mean,Y_t_mean ]).reshape(2,-1)
        
        #MeanCentered
        X_t_1MC = Meas_Z_t_1[0,:] - X_t_1mean
        Y_t_1MC = Meas_Z_t_1[1,:] - Y_t_1mean
        t_1MC = np.array([X_t_1MC, Y_t_1MC ]).reshape(2,-1)
        
        X_tMC = Meas_Z_t[0,:] - X_t_mean
        Y_tMC = Meas_Z_t[1,:] - Y_t_mean
        t_MC = np.array([X_tMC, Y_tMC ]).reshape(2,-1)
        #SVD Method
        q_t_1 = t_1MC.T
        q_t = t_MC.T
        H = q_t.T @ q_t_1
        u, s, vh = np.linalg.svd(H,full_matrices=True,compute_uv=True)
        
        #Find Roation
        R = np.matmul(vh.T,u.T)
        if np.linalg.det(R) <= -1:
            Warning('No Unique solution obtained')  
            
        T = t_1mean - R@t_mean 
        # error = np.sum(np.hypot(X_t_1MC , Y_t_1MC)**2) - 2*np.sum(s)
        
        error = np.sum( np.square(t_mean - R@t_1mean))
        
        orientation = np.rad2deg(np.arctan2(R[1,0],R[0,0]))
        if np.any(np.isnan(T)):
            print('nan')
        return R,T,orientation,error
    
    def match(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 10,threshold = 0.0001):
        Meas_Z_t = Meas_Z_t[[0,1],:]
        Meas_Z_t_1 = Meas_Z_t_1[[0,1],:]
        if Meas_Z_t.shape != Meas_Z_t_1.shape:
            lt = min(Meas_Z_t_1.shape[1], Meas_Z_t.shape[1] )
            Meas_Z_t_1 = Meas_Z_t_1[:,0:lt]
            Meas_Z_t = Meas_Z_t[:,0:lt]      
            
        prev_error = 0
        dist_error = 10000
        loop_cnt = 0
        actMeas = Meas_Z_t.copy()
        #Transform to the estimated pose
        Meas_Z_t = rotate(Est_X_t[2]) @ Meas_Z_t + Est_X_t[0:2].reshape(2,-1)
        Meas_Z_t_1 = rotate(Est_X_t_1[2]) @ Meas_Z_t_1 + np.array([Est_X_t_1[0], Est_X_t_1[1]]).reshape(2,-1)
        try:
            while np.abs(prev_error - dist_error) > threshold:
                prev_error = dist_error
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Meas_Z_t.T)
                dist, Colidx = nbrs.kneighbors(Meas_Z_t_1.T)
                # if np.mean(dist)>5:
                #     print(np.mean(dist))
                Colidx = Colidx[dist< min(np.median(dist), 5)] #Use the closest point
                Colidx = np.unique(Colidx)
                R,T,orientation,alignmenterr = self._compute_T_R(Meas_Z_t_1[:,Colidx ],Meas_Z_t[:,Colidx ])
                
                Meas_Z_t = R @ Meas_Z_t  +T
                dist_error = alignmenterr
                loop_cnt +=1
                if dist_error< threshold or loop_cnt > Iter:
                    break
            #print(alignmenterr)    
            # if alignmenterr > 5: #Alignment impossible
            #     #print(f"Error beyond threshold @ {alignmenterr}")
            #     alignmenterr = None
            RelativeTrans = {'r':R,'T':T , 'yaw':orientation,'error':alignmenterr}
            #print(f"Relative Trans: {RelativeTrans}")
            
            R,T,orientation,error = self._compute_T_R(Meas_Z_t,actMeas)
            GlobalTrans = {'r':R,'T':T , 'yaw':orientation,'error':alignmenterr}
            #print(f"Global Trans: {GlobalTrans}")
            # Calculate T and R between actual measurement  and the transformed scan

            return GlobalTrans
        except Exception as e:
            print(e)
            
class RTCSM():
    def __init__(self, og,Est_X_t, Meas_Z_t):
        self.OG = og
        self.searchWindow_t = 4.0 # meters
        self.searchWindow_yaw = 90 #degrees
        self.highResMap_t_1,centre_pos = self._highResMap(Est_X_t, Meas_Z_t)
        #Est_X_t, Meas_Z_t = self._modifyDtype(Est_X_t, Meas_Z_t)      
        self.lowResMap_t_1 = self._lowResMap(self.highResMap_t_1, centre_pos, Est_X_t)
        self.Pos_var = 0.1
        self.ori_var = 0.01
        self.updateTargetMap(Est_X_t, Meas_Z_t)
        
    def match(self, Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1):
        #Construct High resolution map
        HMap,centre_pos = self._highResMap(Est_X_t, Meas_Z_t)
        #Est_X_t, Meas_Z_t = self._modifyDtype(Est_X_t, Meas_Z_t)
        
        #Construct Low resolution Map
        LMap = self._lowResMap(HMap, centre_pos, Est_X_t)
        Updt_X_t, confidence_ = self._search3DMatch( Est_X_t, Est_X_t_1, Meas_Z_t, centre_pos, LMap, 0.5,mode='L')
        Updt_X_t, confidence = self._search3DMatch(Updt_X_t, Est_X_t_1, Meas_Z_t, centre_pos, HMap, 0.1, mode= 'H')
        Trans = {'error':confidence, 'T':np.array([Updt_X_t[0], Updt_X_t[1]]).reshape(2,-1), 'yaw':Updt_X_t[2]}
        return Trans
    
    def _highResMap(self, Est_X_t, Meas_Z_t):
        dim = self.OG.max_lidar_r + self.OG.Roffset
        
        HMap,centre_pos,_,_ = self.OG.getScanMap(Meas_Z_t, Est_X_t)
                
        Xrange = np.arange(centre_pos[0][0]-dim ,  centre_pos[0][0]+dim)
        Yrange = np.arange(centre_pos[1][0]-dim ,  centre_pos[1][0]+dim)     
        
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
        max_range = self.OG.max_lidar_r
        MeasMap = Map
        # create a search space
        Numcell = (max_range+ self.OG.Roffset)*searchStep
        Xrange = np.arange(-Numcell ,  Numcell, searchStep)
        Yrange = np.arange(-Numcell ,  Numcell, searchStep)
        x , y = np.meshgrid(Xrange,Yrange)
        if mode == 'H':
            TargetMap = self.HTgtMap
            ori_space = np.arange(-1,1,searchStep)
            Pos_search_mask = np.zeros((Map.shape[0], Map.shape[1]))
            Ori_search_mask = np.zeros((Map.shape[0], Map.shape[1]))
        else:
            TargetMap = self.HTgtMap
            ori_space = np.arange(-5,5,searchStep)
        
            EstDistDrvn = np.sqrt((Est_X_t_1[0] - Est_X_t[0])**2 + (Est_X_t_1[1] - Est_X_t[1])**2 )
            sq = np.array((x)**2 + (y)**2, dtype=np.float32)
            sq[np.where(sq ==0)] = np.ceil(EstDistDrvn)
            Pos_search_mask = (-(1 / (2 * self.Pos_var**2)) * (np.sqrt( np.abs(sq - EstDistDrvn))))#*0.0000001
            Pos_search_mask[Pos_search_mask > 0.5] = -100

            distv = np.sqrt(x**2 + y**2)
            distv[distv == 0] = 0.0001
            yaw = np.deg2rad(Est_X_t[2]  - Est_X_t_1[2])
            Ori_search_mask = np.arccos((x* np.cos(yaw) + y* np.sin(yaw)) / distv)
            Ori_search_mask = -1 / (2 * self.ori_var ** 2) * np.square(Ori_search_mask)        
        
        Corr_Cost_Fn = np.zeros((len(ori_space),Map.shape[0],Map.shape[0]))
        Corr_cost = np.zeros(len(ori_space),)
        theta = np.zeros(len(ori_space),)
        for i,ori in enumerate(ori_space):
            T = TargetMap.copy()
            Est= [Est_X_t[0],Est_X_t[1] ,Est_X_t[2]+ori]
            #Meas = self._rotate(Est_X, Meas_Z_t)
            #Temp_map, centre_pos,_,_ = self._highResMap(Est, Meas_Z_t)      
            Temp_map= sp.ndimage.rotate(MeasMap, ori , reshape=False)
            # self.OG.PlotMap(Temp_map,Est,'Temp_map')     
            #self.OG.PlotMap(np.rot90(Temp_map,0),Est,'tmp map')
            # plt.contour(Pos_search_mask)
            # plt.contour(Ori_search_mask)
            #plt.show()
            if Temp_map.shape != Pos_search_mask.shape:
                Pd_sz = round(Temp_map.shape[0]/2) - round(Pos_search_mask.shape[0]/2)
                Pos_search_mask = np.pad(Pos_search_mask, (Pd_sz, Pd_sz), 'constant', constant_values=(0,0))
                Ori_search_mask = np.pad(Ori_search_mask, (Pd_sz, Pd_sz), 'constant', constant_values=(0,0))
                
            UncrtnMap = Temp_map + Pos_search_mask +Ori_search_mask
            theta[i] = Est[2]
            Corr_cost[i] = norm(np.abs(T - UncrtnMap), ord=1)
            Corr_Cost_Fn[i,:,:] = T - UncrtnMap
            
        # Find the best voxel in the Low resolution Map
        minIdx = np.unravel_index(Corr_Cost_Fn.argmin(), Corr_Cost_Fn.shape)
        confidence = min(1/Corr_cost)
        dx, dy, dtheta = Xrange[minIdx[1]] , Yrange[minIdx[2]], ori_space[minIdx[0]]
        Updt_X_t = [Est_X_t_1[0] + dx, Est_X_t_1[1] + dy, Est_X_t_1[2] + dtheta]
        # print(dx,dy,dtheta)
        return Updt_X_t , confidence
        
    def _rotate(self, Est_X, Meas):
        RotatedMeas ={}
        Meas = np.array([Meas['x'], Meas['y']]).reshape(2,-1)
        Meas = rotate(Est_X['yaw'])@ Meas 
        RotatedMeas['x'] = Meas[0,:]
        RotatedMeas['y'] = Meas[1,:]
        RotatedMeas['Azimuth'] = np.rad2deg(np.arctan2(RotatedMeas['y'] , RotatedMeas['x']))
        RotatedMeas['Range_XY_plane']= np.hypot(RotatedMeas['x'],RotatedMeas['y'])
        return RotatedMeas
    
    def updateTargetMap(self, Est_X_t, Meas_Z_t):
        self.Est_X_t_1 = Est_X_t
        self.Meas_Z_t_1 = Meas_Z_t  
        ExtractMap,_= self.OG.getExtractMap(Est_X_t)
        self.HTgtMap =  gaussian_filter(ExtractMap, sigma =1)
        self.LTgtMap =  gaussian_filter(ExtractMap, sigma =3)
        
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
    Meas_Z_t_1 =np.array([[1,-1],[1,1]],dtype = np.float64)
    Meas_Z_t = np.array([[1,-1],[0,2]],dtype = np.float64)
    Est_X_t = {'x': 0 , 'y':2 , 'yaw':90 } # np.array([0,0,0],dtype = np.float64)
    Est_X_t_1 ={'x': 0 , 'y':2 , 'yaw':0 } # np.array([0,2,0],dtype = np.float64)
    SM = ICP()   
    R,T,orientation,error = SM.match(Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1)