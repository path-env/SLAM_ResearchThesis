# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 09:59:08 2020

@author: MangalDeep
"""
import numpy as np
import sys
from numpy.core.fromnumeric import shape
from numpy.core.numeric import _move_axis_to_0, indices
from numpy.lib.stride_tricks import _maybe_view_as_subclass
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.linalg import norm
import scipy as sp
import matplotlib.pyplot as plt
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as normz
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
    def __init__(self, OG):
        self.KernelSize = 10
        self.logger = logging.getLogger('ROS_Decode.SM_ND')
        self.logger.info("ScanMatching Initialized")
        self.ICP = ICP()
        self.OG = OG

    def CreateNDMap(self,OG,Pose):
        #Grab the map constructed so far and construct MAP ND
        self.Pose = Pose
        self.OG = OG
        self.Map_ND_k_1_G = self.Map_ND()
    
    def match(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1):
        self.logger.info("Matching the scan with current estimate to the Global Map")
        ##Robot Coordinate Frame
        #ICP to derive R to R_1
        RT, pk_ICP =  self.ICP.match_SVD(Meas_Z_t_1,Meas_Z_t,Est_X_t,Est_X_t_1, Mode = 'RT')
        Meas = Meas_Z_t[:2,:]        
        Z_k_R_1 = rotate(pk_ICP[2]) @ Meas + RT['T']
        Meas_R_1 = {'x': Z_k_R_1[0,:] ,'y': Z_k_R_1[1,:]}
        ## Global Coordinate system
        #In Global Frame
        Z_k_G = rotate(Est_X_t_1[2]) @ Z_k_R_1 + translate(Est_X_t_1[0], Est_X_t_1[1])
        #Compute Robot pose in Global for time K     
        Est_X_k_G = rotate(Est_X_t_1[2]) @ RT['T'] + translate(Est_X_t_1[0],Est_X_t_1[1])
        Est_phi_k_G = RT['yaw'] + Est_X_t_1[2]

        RT, pk_ICP =  self.ICP.match_SVD(Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1)
        Z_k_G = rotate(pk_ICP[2]) @ Meas + translate(pk_ICP[0], pk_ICP[1])
        ## Corrected global coordinate system
        #Transform Scan to GCS
        # Meas_Z_G = rotate(Est_phi_k_G)@ Z_k_G + translate(Est_X_k_G[0],Est_X_k_G[1]) 
        # Estimate the corrected global coordinate values
        pk_Gplus = self.correctedGplus(Meas, Est_X_t)
        #In Global Frame
        Z_k_Gplus = rotate(pk_Gplus[2]) @ Z_k_G + translate(pk_Gplus[0], pk_Gplus[1])
        

        # Scan_ND_k_G, ScanMap = self.Scan_ND(Meas_Z_G,Est_X_t) 
        
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
                
    def Map_to_GCS(self,Map,Pose_X_t = {'x':0 , 'y':0 , 'yaw':0}):
        #Only coordinates are processed not the entire matrix
        Map_G = [0,1]
        Global_CG = self.OG.getMapPivotPoint()        
        Map_G[0] =    Global_CG[0] - Map[0]
        Map_G[1] =    Global_CG[1] - Map[1]
        return Map_G[0] , Map_G[1]    

    def correctedGplus(self, Meas, Est_X_t):
        # Meas_Z_G = Meas_Z_G + self.OG.MapDim
        # Meas_Z_G[1,:] = Meas_Z_G[1,:] + 100
        # Map_ND =self.getMap_ND()
        ScanMap,_,_,_ =self.OG.getScanMap(Meas,Est_X_t)
        _, Meas_Z_G = self.OG.Lidar2MapFrame(ScanMap , Est_X_t)
        Map_ND_k_1_G = self.getOccupancyInfo2(self.OG.MapIdx_G, Meas_Z_G)
        # Scan_ND = self.getScan_ND(Meas_Z_G)
        # 
        pass

    def getMap_ND(self):   
        self.logger.info("Create ND Global Map")        
        # OccVal = self.OG.LO_t_i
        # GMap = self.OG.LO_t_i
        Map_ND_k_1_G = self.getOccupancyInfo2(self.OG.MapIdx_G)
        #self.OG.PlotMap(np.rot90(self.ND_map,3),self.Pose,'Global ND Map')
        #self.PlotMapND(self.ND_map,'Global')
        return Map_ND_k_1_G

    def getScan_ND(self,Meas_Z_G):
        self.logger.info("Create ND Scan Map")
        Scan_ND_k_G = self.getOccupancyInfo2(Meas_Z_G)
        return Scan_ND_k_G 
               
    def getOccupancyInfo2(self,MapOccIdx, ScanOccIdx):
        OccVal = self.OG.l_occ
        KS = self.KernelSize
        #Appply on Map recursively creating cells j
        corres = []
        #ND = np.zeros(Map.shape)
        #OccIdx = self.OG.MapIdx_G 
        Map_NDS_mu, Scan_NDS_mu = [], []
        Map_NDS_sig, Scan_NDS_sig = [], []
        for row in range(0,self.OG.Long_Length, KS):
            Idx1_1 = np.where(np.logical_and(MapOccIdx[0,:]>row,MapOccIdx[0,:]<KS+row))
            Idx1_2 = np.where(np.logical_and(ScanOccIdx[0,:]>row,ScanOccIdx[0,:]<KS+row))
            if Idx1_1[0].__len__() ==0 or  Idx1_2[0].__len__() ==0:
                continue
            for col in range(0,self.OG.Lat_Width , KS):
                Idx2_1 = np.where(np.logical_and(MapOccIdx[1,:]>col,MapOccIdx[1,:]<KS+col))
                Idx2_2 = np.where(np.logical_and(ScanOccIdx[1,:]>col,ScanOccIdx[1,:]<KS+col))
                if Idx2_1[0].__len__() ==0 or Idx2_2[0].__len__() ==0:
                    continue
                Idx1 = np.intersect1d(Idx1_1, Idx2_1)
                Idx2 = np.intersect1d(Idx1_2, Idx2_2)
                if Idx1.__len__() ==0 or Idx2.__len__() ==0:
                    continue
                SlctIdx1 = MapOccIdx[:,Idx1]
                SlctIdx2 = ScanOccIdx[:,Idx2]
                Map_ND = self.getSurfaces(SlctIdx1)
                Scan_ND = {'mu': SlctIdx2.mean(axis =1) ,'sig': np.cov(SlctIdx2, fweights=np.ones(SlctIdx2.shape[1])*10 )}
                # Scan_ND = self.getSurfaces(SlctIdx2)
                Map_ND, KL = self.div_KullbackLeibler(Map_ND, Scan_ND)
                corres.append(KL)
                Map_NDS_mu.append(Map_ND['mu'])
                Map_NDS_sig.append(Map_ND['sig'])
                Scan_NDS_mu.append(Scan_ND['mu'])
                Scan_NDS_sig.append(Scan_ND['sig'])
        self.getGPlus(Map_NDS_mu, Map_NDS_sig,Scan_NDS_mu,Scan_NDS_sig,corres)
        return Map_NDS
    
    def div_KullbackLeibler(self, Map_ND, Scan_ND):
        KL = []
        for M_ND in Map_ND:
            # Covariance
            p1 = np.linalg.pinv(M_ND['sig'])
            p2 = Scan_ND['sig']
            # mu
            p3 = M_ND['mu']
            p4 = Scan_ND['mu']
            tmp = (p3 - p4).T @ p1 @ (p3 - p4)
            p5 = np.log(max(np.linalg.det(p2),0.0000001)/max(np.linalg.det(M_ND['sig']),0.0000001))
            dim = 2
            ans = -0.5*(np.trace(p1 @ p2) + tmp - p5 - dim)
            KL.append(ans)
        min_l = np.argmin(KL)
        return Map_ND[min_l], KL[min_l]
        '''
                Tmp = {}
                Tmp['j'] = jj
                map_extract = Map[row:KS+row,col:KS+col]
                # ND_kernel= gaussian_filter(map_extract, sigma = 2) ## Only a representation 
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
                '''
            
    def Cell_Correspondence(self, Map_NDs, Scan_NDs):
        MapIdx = np.where(*Map_NDs.keys() >1 )
        ScanIdx = np.where(*Scan_NDs.keys() >1 )
        return np.intersect1d(MapIdx, ScanIdx)
    
    def getGPlus(self,Map_NDS_mu, Map_NDS_sig,Scan_NDS_mu,Scan_NDS_sig,corres):
        Map_NDS_mu,Map_NDS_sig = np.array(Map_NDS_mu), np.array(Map_NDS_sig)
        Scan_NDS_mu,Scan_NDS_sig = np.array(Scan_NDS_mu), np.array(Scan_NDS_sig)
        p = np.array([0,0,0])
        ScanGp_ND = []
        for _ in range(10):
            Scan_NDS_mu = rotate(p[2]) @ Scan_NDS_mu.T + p[0:2].reshape(2,1)
            Scan_NDS_sig = rotate(p[2]) @ Scan_NDS_sig @ np.linalg.inv(rotate(p[2]))

            J = np.zeros((2,3,Map_NDS_mu.shape[0]))
            H = np.zeros((3,3))
            g = np.zeros((3,1))
            t = Map_NDS_mu.T - Scan_NDS_mu
            deg_x = np.rad2deg(p[2])
            J[0,0,:] , J[1,1,:] =1,1
            J[0,2,:] = -np.sin(deg_x)*t[0,:] - np.cos(deg_x)*t[1,:] 
            J[1,2,:] = np.cos(deg_x)*t[0,:]  + np.sin(deg_x)*t[1,:]
            for j in range(Map_NDS_mu.shape[0]):
                tmp = t[:, j] @ np.linalg.pinv(Scan_NDS_sig[j,:,:]) @ J[:,:,j]
                H += tmp @ tmp.T
                g +=  tmp @ corres[j]
                chi2 += e[:,j:j+1].T @e[:,j:j+1]
            # print(f"The chi^2 error:{chi2}, matshape:{len(indices)}")
            chi2 = chi2/p1.shape[1]                   
        pass

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
               
    def getSurfaces(self, Idx):
        Idx_df = pd.DataFrame(Idx.T, columns=['x', 'y'])
        Idx_x = Idx_df.apply(np.floor).groupby("x", axis=0)
        Idx_y = Idx_df.apply(np.floor).groupby("y", axis=0)
        NDs = []
        if Idx_x.__len__() > Idx_y.__len__():
            # Group based on Y
            for i,l in enumerate(Idx_y.indices.keys()):
                index = Idx_y.indices[l]
                mu = Idx[:,index].mean(axis =1)
                sig = np.cov(Idx[:,index], fweights=np.ones(index.shape[0])*10 )
                tmp = {'l':l, 'mu': mu ,'sig': sig}
                NDs.append(tmp)
        elif Idx_x.__len__() <= Idx_y.__len__():
            for i,l in enumerate(Idx_x.indices.keys()):
                index = Idx_x.indices[l]
                mu = Idx[:,index].mean(axis =1)
                sig = np.cov(Idx[:,index], fweights=np.ones(index.shape[0])*10)
                tmp = {'l':l, 'mu': mu ,'sig': sig}
                NDs.append(tmp)
        else:
            pass
        return NDs

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
        R = u @ vh
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
        p1_Idx,_ = np.where(dist<= tolerance) # ICP LS = 0.5,0.1
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
        # Meas_Z_t = np.round(Meas_Z_t,5)
        Meas_Z_t = np.unique(Meas_Z_t, axis=1)
        # Meas_Z_t_1 = np.round(Meas_Z_t_1,5)
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

    def match_SVD(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 100,threshold = 0.000001, Mode = 'GT'):
        Meas_Z_t = Meas_Z_t[[0,1],:]
        Meas_Z_t_1 = Meas_Z_t_1[[0,1],:]
        # if Meas_Z_t.shape != Meas_Z_t_1.shape:
        #     lt = min(Meas_Z_t_1.shape[1], Meas_Z_t.shape[1] )
        #     Meas_Z_t_1 = Meas_Z_t_1[:,0:lt]
        #     Meas_Z_t = Meas_Z_t[:,0:lt]      
        prev_error,bound = 0, 1
        alignmenterr = 10000
        loop_cnt = 0
        # Transform to the estimated pose
        actMeas = Meas_Z_t_1.copy()
        # Meas_Z_t = rotate(Est_X_t[2]) @ Meas_Z_t + Est_X_t[0:2].reshape(2,-1)
        # Meas_Z_t_1 = rotate(Est_X_t_1[2]) @ Meas_Z_t_1 + Est_X_t_1[0:2].reshape(2,-1)
        #Transform to the estimated relative pose
        diff= Est_X_t - Est_X_t_1
        Meas_Z_t_1 = rotate(diff[2]) @Meas_Z_t_1 + diff[0:2].reshape(2,-1) 
        try:
            while np.any(np.abs(prev_error - alignmenterr) > threshold):
                prev_error = alignmenterr
                # bound = max(0.1,bound*np.exp(-0.1*loop_cnt))
                # Meas_Z_t, Meas_Z_t_1 = self.subsample(Meas_Z_t, Meas_Z_t_1)
                p2,p1, weights =self._closestPtCorrespondence(Meas_Z_t, Meas_Z_t_1,bound)
                R,T,orientation,alignmenterr,_,_,_,_,p_1mean,p_mean = self._compute_T_R(p2,p1, weights)                
                Meas_Z_t_1 = (R @ (Meas_Z_t_1[:,:]- p_mean)) + p_1mean # '- p_1mean' for standalone run
                loop_cnt +=1
                # print(alignmenterr, orientation)
                if np.all(abs(alignmenterr)< threshold) or loop_cnt > Iter:
                    break   
            # if alignmenterr > 5: #Alignment impossible
            #     #print(f"Error beyond threshold @ {alignmenterr}")
            #     alignmenterr = None
            RelativeTrans = {'r':R,'T':T , 'yaw':orientation,'error':alignmenterr}
            R,T,orientation,error,_,_,_,_,_,_ = self._compute_T_R(actMeas,Meas_Z_t_1, weights)# swap act and t-1 for standalone run
            Trans_Lst = np.append(T,orientation)
            if Mode == 'GT':
                Trans_Lst = poseComposition(Est_X_t_1, Trans_Lst)
            Trans_Dict = {'r':rotate(Trans_Lst[2]),'T':Trans_Lst[:2].reshape(2,1) , 'yaw':Trans_Lst[2],'error':np.linalg.norm(error)}
            return Trans_Dict,Trans_Lst
        except Exception as e:
            print(e)

    def match_LS(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 100,threshold = 0.000001, Mode = 'GT'):
        # Using Gausss Newton approximation
        chi2_lst, X_lst = [], []
        bound = 3
        del_x = np.array([0.,0.,0.]).reshape(3,-1)
        # del_x = (Est_X_t[:3] - Est_X_t_1[:3]).reshape(3,-1)
        # del_x[2] = np.deg2rad(del_x[2])
        init_est = del_x.copy()
        x =  np.array([0.,0.,0.]).reshape(3,-1)
        diff= Est_X_t - Est_X_t_1
        P2, P1 =(Meas_Z_t[0:2,:], Meas_Z_t_1[0:2,:])
        # P2, P1 = self.subsample(p2, p1)
        P2 = rotate(Est_X_t[2])@P2 + Est_X_t[0:2].reshape(2,-1)
        P1 = rotate(Est_X_t_1[2])@P1 + Est_X_t_1[0:2].reshape(2,-1)
        P1 = rotate(diff[2]) @P1[0:2,:] + diff[0:2].reshape(2,-1) 
        P1_org = P1.copy()   
        for t in range(Iter):
            chi2,deg_x = 0, np.rad2deg(x[2])
            # Apply transformation on pointcloud
            P1 = rotate(deg_x)@P1_org + x[0:2]
            # bound = max(0.1,bound*np.exp(-0.1*t))
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
            J[0,2,:] = -np.sin(x[2])*p1[0,:] - np.cos(x[2])*p1[1,:] 
            J[1,2,:] = np.cos(x[2])*p1[0,:]  - np.sin(x[2])*p1[1,:] 
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
            chi2_lst.append(chi2.flatten())
            X_lst.append(x.copy())
        X_lst = np.array(X_lst)
        minIndx,_ = np.where(chi2_lst==min(chi2_lst))
        Trans_Lst = X_lst[minIndx-1][0].flatten()
        Trans_Lst[2] = np.rad2deg(Trans_Lst[2])
        if Mode == 'GT':
            Trans_Lst = poseComposition(Est_X_t_1, Trans_Lst)   
        error = max(min(chi2_lst),0.00001)
        Trans_Dict = {'r':rotate(Trans_Lst[2]),'T': Trans_Lst[:2].reshape(2,1), 'yaw':Trans_Lst[2],'error':error}
        return Trans_Dict,Trans_Lst
    
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
        # self.highResMap_t_1 = self._highResMap(Est_X_t, Meas_Z_t)
        #Est_X_t, Meas_Z_t = self._modifyDtype(Est_X_t, Meas_Z_t)      
        # self.lowResMap_t_1 = self._lowResMap(self.highResMap_t_1, Est_X_t)
        self.updateTargetMap(Est_X_t, Meas_Z_t)
        self.Pos_var = 0.1
        self.ori_var = 0.1
        
    def match(self, Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1, iteration):
        self.iteration = iteration
        #Construct High resolution map
        HMap = self._highResMap(Est_X_t, Meas_Z_t)
        #Est_X_t, Meas_Z_t = self._modifyDtype(Est_X_t, Meas_Z_t)
        
        #Construct Low resolution Map
        LMap = self._lowResMap(HMap, Est_X_t)
        Updt_X_t, confidence_ = self._search3DMatch( Est_X_t, Est_X_t_1, Meas_Z_t,  LMap, 0.1,mode='L')
        Updt_X_t, confidence = self._search3DMatch(Updt_X_t, Est_X_t_1, Meas_Z_t,  HMap, 0.01, mode= 'H')
        Trans = {'error':confidence, 'T':np.array([Updt_X_t[0], Updt_X_t[1]]).reshape(2,-1), 'yaw':Updt_X_t[2]}
        return Trans, Updt_X_t
    
    def _highResMap(self, Est_X_t, Meas_Z_t):        
        # HMap,centre_pos,_,_ = self.OG.getScanMap(Meas_Z_t, Est_X_t)
        # Xrange = np.arange(centre_pos[0][0]-self.dim ,  centre_pos[0][0]+self.dim)
        # Yrange = np.arange(centre_pos[1][0]-self.dim ,  centre_pos[1][0]+self.dim)     
        
        # Map_confined = HMap[Yrange[0]:Yrange[-1]+1, Xrange[0]:Xrange[-1]+1]
        Map_confined,_,_,_ = self.OG.getScanMap(Meas_Z_t, Est_X_t)
        # dist_to_grid = dist_to_grid[Xrange[0]:Xrange[-1], Yrange[0]:Yrange[-1]]
        #self.OG.PlotMap(np.rot90(Map_confined,3),Est_X_t,'High resolution Map')
        return Map_confined

    def _lowResMap(self,HMap, Est_X_t):
        LowResol = 2#np.int32(self.OG.Grid_resol/0.01)
        Map_X_len, Map_Y_Len = HMap.shape
        LowResolMap = np.zeros(HMap.shape)
        for row in range(0,Map_X_len,LowResol):
            for col in range(0,Map_Y_Len,LowResol):
                LowResolMap[row:row+LowResol, col:col+LowResol] = np.max(HMap[row:row+LowResol, col:col+LowResol])
        #self.OG.PlotMap(LowResolMap,Est_X_t,'Low resolution Map 3:1',7000,7000)         
        return LowResolMap
    
    def _search3DMatch(self, Est_X_t, Est_X_t_1, Meas_Z_t,  Map, searchStep,mode= 'L'):                
        MeasMap = Map
        # create a search space
        if mode == 'L':
            Numcell = 20 #0.1
        else:
            Numcell = 10 #0.01
            #Numcell = 15
        Xrange = np.flip(np.arange(-Numcell ,  Numcell+1))
        Yrange = np.flip(np.arange(-Numcell ,  Numcell+1))
        y,x = np.meshgrid(Xrange,Yrange)
        if mode == 'H':
            TargetMap = self.HTgtMap.T
            ori_space = (np.arange(-0.1,0.1,searchStep))
            Pos_search_mask = np.zeros((x.shape[0], x.shape[1]))
            Ori_search_mask = np.zeros((x.shape[0], x.shape[1]))
        else:
            TargetMap = self.LTgtMap.T
            ori_space = (np.arange(-1,1,searchStep))
        
            EstDistDrvn = np.sqrt((Est_X_t_1[0] - Est_X_t[0])**2 + (Est_X_t_1[1] - Est_X_t[1])**2 )
            Pos_contour = np.abs(np.sqrt((x*searchStep) ** 2 + (y*searchStep) ** 2) - EstDistDrvn)
            # sq[np.where(sq <EstDistDrvn)] = np.ceil(EstDistDrvn)
            Pos_search_mask = (-(1 / (2 * self.Pos_var**2)) * (np.sqrt((x*searchStep) ** 2 + (y*searchStep) ** 2) - EstDistDrvn) ** 2)
            # Pos_search_mask = normalize(Pos_search_mask)
            Pos_search_mask[Pos_contour>1] = -100
            #Pos_search_mask[Pos_search_mask < -25000] = -10000000
            if np.abs(Est_X_t[2]  - Est_X_t_1[2]) > 1:
                distMap = np.sqrt(x**2 + y**2)
                distMap[distMap == 0] = 0.0001
                yaw = np.deg2rad(Est_X_t[2]  - Est_X_t_1[2])
                Ori_search_mask = np.arccos((x* np.cos(yaw) + y* np.sin(yaw)) / distMap) 
                Ori_search_mask = -1 / (2 * self.ori_var ** 2) * np.square(Ori_search_mask)
            else:
                Ori_search_mask = np.zeros((x.shape[0], x.shape[1]))
            # Pos_search_mask = np.zeros((Map.shape[0], Map.shape[1]))
            # Ori_search_mask = np.zeros((Map.shape[0], Map.shape[1]))
        Mask = Pos_search_mask + Ori_search_mask
        # Mask = Mask.reshape((Mask.shape[0], Mask.shape[1], 1))
        #Mask = normalize(Mask)
        #Mask[Mask<-0.6] = 0
        x = x.reshape((x.shape[0], x.shape[1], 1))
        y = y.reshape((y.shape[0], y.shape[1], 1))
        Corr_Cost_Fn = np.zeros((len(ori_space), x.shape[0],x.shape[1]))
        Corr_cost = np.zeros(len(ori_space),)
        theta = np.zeros(len(ori_space),)
        #Meas['x'], Meas['y']= Meas_Z_t['x'], Meas_Z_t['y']
        Est =[Est_X_t[0],Est_X_t[1]+100,Est_X_t[2]]
        resol = 1/0.01
        # plt.figure()
        for i,ori in enumerate(ori_space):
            Est[2] = Est_X_t[2] + ori
            #Meas = self._rotate(Est, Meas, searchStep)
            Temp_map = sp.ndimage.rotate(MeasMap, ori , reshape=False, order=1)
            Temp_map_G,_= self.OG.Lidar2MapFrame(Temp_map , Est_X_t)
            Temp_map_G[Temp_map_G<1.5] = 0
            # m = self.rotate(Est,Meas)
            MeasIdx = np.asarray(np.where(Temp_map_G.T>0))
            XY_Idx = np.unique(MeasIdx, axis=1)
            X_Idx, Y_Idx = XY_Idx[0,:],XY_Idx[1,:]
            # _,Idx = np.unique(XY_Idx[0,:],return_index=True)
            # XY_Idx = XY_Idx[:,Idx]
            # _,Idx = np.unique(XY_Idx[1,:],return_index=True)
            # X_Idx, Y_Idx = XY_Idx[0,:], XY_Idx[1,:]

            # Idx = np.where(np.logical_and(X_Idx<65 , X_Idx>5))
            # X_Idx, Y_Idx = X_Idx[Idx], Y_Idx[Idx]
            # Idx = np.where(np.logical_and(Y_Idx<65 , Y_Idx>5))
            # X_Idx, Y_Idx = X_Idx[Idx], Y_Idx[Idx]

            # TMap = np.zeros(TargetMap.shape)
            # TMap[X_Idx, Y_Idx] = Temp_map[X_Idx, Y_Idx]
            # TMap = TargetMap[X_Idx.min():X_Idx.max(), Y_Idx.min(), Y_Idx.max()] 
            X_Idx = X_Idx.reshape(1, 1, -1) #- self.OG.max_lidar_r
            Y_Idx = Y_Idx.reshape(1, 1, -1) #- self.OG.max_lidar_r
            # Meas = (rotate(Est[2]+90) @ Meas_Z_t[0:2,:]) + translate(Est[0], Est[1]) + self.OG.MapDim
            X_Idx = (X_Idx + x).astype(int)
            Y_Idx = (Y_Idx + y).astype(int)
            convResult = (TargetMap[X_Idx, Y_Idx]) #+Mask.reshape(Mask.shape[0], Mask.shape[1],1)
            convResultSum = np.sum(convResult, axis=2)
            # convResultSum = normalize(convResultSum)
            UncrtnMap = convResultSum + Mask
            theta[i] = Est[2]
            Corr_cost[i] = MeasIdx.shape[1]
            Corr_Cost_Fn[i,:,:] = UncrtnMap
            # Debug
            # plt.cla()
            # self.OG.PlotMap(Temp_map_G,Est,'Temp_map_G',self.OG.Long_Length,self.OG.Lat_Width)   
            # # plt.contour(Pos_search_mask)
            # # plt.contour(Ori_search_mask)
            # plt.contourf(UncrtnMap)
            # # plt.colorbar()
            # plt.pause(0.001)
        # Find the best voxel in the Low resolution Map
        # plt.close()
        maxIdx = np.unravel_index(Corr_Cost_Fn.argmax(), Corr_Cost_Fn.shape)
        temp = (Corr_Cost_Fn[maxIdx[0],:,:])
        temp = (temp - temp.min())/(temp.max()-temp.min())
        temp = temp/temp.sum()
        confidence = temp.max()
        dx, dy, dtheta = Xrange[maxIdx[2]]*searchStep , Yrange[maxIdx[1]]*searchStep, ori_space[maxIdx[0]]
        print(maxIdx)
        # if confidence>0.056:
        #     print("cx")
        Updt_X_t = [Est_X_t[0] + dx, Est_X_t[1] + dy, Est_X_t[2]+dtheta]
        # print(TargetMap.max())
        # print(f'Mode = {mode},Est = {Est}, Updt = {Updt_X_t}')
        # if (confidence) < 0.5:
        print(confidence)
        return Updt_X_t , 1/confidence
        
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
        # ExtractMap,_= self.OG.getExtractMap(Est_X_t)
        # ExtractMap ,_= self.OG.Lidar2MapFrame( ExtractMap.T, Est_X_t) 
        ExtractMap = self.OG.LO_t_i
        ExtractMap[ExtractMap<1.5] = 0
        # ExtractMap[ExtractMap>0] = 1
        self.HTgtMap =  ExtractMap #gaussian_filter(ExtractMap, sigma =1)
        self.LTgtMap =  ExtractMap #gaussian_filter(ExtractMap, sigma =1)
        # LowResol = np.int32(3*self.OG.Grid_resol/0.01)
        # LowResolMap = np.zeros((ExtractMap.shape))
        # for row in range(0,ExtractMap.shape[0],LowResol):
        #     for col in range(0,ExtractMap.shape[1],LowResol):
        #         LowResolMap[row:row+LowResol, col:col+LowResol] = gaussian_filter(ExtractMap[row:row+LowResol, col:col+LowResol], sigma=30)
        # self.LTgtMap =  LowResolMap
        
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
    R = 2.2 # in degress
    T = np.array([1,-1.5]).reshape(2,1)
    Meas_Z_t_1 =np.array([[1,-1],[1,1]],dtype = np.float64)
    Meas_Z_t = np.array([[1,-1],[0,2]],dtype = np.float64)
    Meas_Z_t_1 = np.array([[2,3,-2,-3,-2,-3,2,3],[2,3,2,3,-2,-3,-2,-3]])
    Meas_Z_t = (rotate(R) @ Meas_Z_t_1 ) + T
    Est_X_t = np.array([1,0.,0.5],dtype = np.float64)
    Est_X_t_1 =np.array([0,0,0],dtype = np.float64)
    SM = ICP()   
    GT, GT_Lst = SM.match_LS(Meas_Z_t_1,Meas_Z_t,Est_X_t,Est_X_t_1)
    print(GT_Lst)