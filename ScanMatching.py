# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 09:59:08 2020

@author: MangalDeep
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging
from sklearn.neighbors import NearestNeighbors

# lambda Functions
rotate = lambda phi: np.array([[np.cos(np.deg2rad(phi)), -np.sin(np.deg2rad(phi))],
                                [np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))]]).reshape(2,2)
translate = lambda x,y : np.array([x , y]).reshape(2,1)

def CoordTrans(Meas,X_t, X_t_1,TargetFrame):
    if TargetFrame == 'G':
        X_t_1 =  {'x':0 , 'y':0 , 'yaw':0}
    #elif TargetFrame == 'R_':
    Meas_Trans = {}
    Z = Meas.T
    dx = X_t['x'] - X_t_1['x']
    dy = X_t['y'] - X_t_1['y']
    dyaw =np.deg2rad(X_t['yaw'] - X_t_1['yaw'])
    Meas_mat = rotate(dyaw) @ Z + translate(dx, dy)
    Meas_Trans['x'] = Meas_mat[0,:]
    Meas_Trans['y'] = Meas_mat[1,:]
    Meas_Trans['Azimuth'] =np.rad2deg(np.arctan2(Meas_Trans['y'] , Meas_Trans['x']))
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
    def __init__(self):
        pass

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
        return R,T,orientation,error
    
    def match(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1,Iter = 10,threshold = 0.01):
        Meas_Z_t = Meas_Z_t[[0,1],:]
        Meas_Z_t_1 = Meas_Z_t_1[[0,1],:]
        if Meas_Z_t.shape != Meas_Z_t_1.shape:
            lt = min(Meas_Z_t_1.shape[1], Meas_Z_t.shape[1] )
            Meas_Z_t_1 = Meas_Z_t_1[:,0:lt]
            Meas_Z_t = Meas_Z_t[:,0:lt]      
            
        prev_error = 0
        dist_error = 10000
        loop_cnt = 0
        OldMeas = Meas_Z_t.copy()
        Meas_Z_t = rotate(Est_X_t['yaw']) @ Meas_Z_t + np.array([Est_X_t['x'], Est_X_t['y']]).reshape(2,-1)
        # Align scans
        while np.abs(prev_error - dist_error) > threshold:
            prev_error = dist_error
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Meas_Z_t.T)
            dist, Colidx = nbrs.kneighbors(Meas_Z_t_1.T)

            Colidx = Colidx[dist<5] #Use the closest point
            Colidx = np.unique(Colidx)
            R,T,orientation,error = self._compute_T_R(Meas_Z_t_1[:,Colidx ],Meas_Z_t[:,Colidx ])
            
            Meas_Z_t = R @ Meas_Z_t  +T
            
            dist_error = np.mean(dist)
            loop_cnt +=1
            if dist_error< threshold or loop_cnt > Iter:
                break
            
        # Calculate T and R between old measurement  and the transformed scan    
        R,T,orientation,error = self._compute_T_R(Meas_Z_t,OldMeas)
        return R,T.flatten(),orientation,error
  
class RTCSM():
    def __init__(self,MapObj):
        self.OG = MapObj
        self.Est_X_t_1 = []  #previosusly matched location
    
    def MultiLvlResol(self, Meas_Z_t, LowRes,HighRes):
        #Low Resolution
        GridSize = LowRes * self.OG.Grid_resol
        xRangeList, yRangeList, probSP = self.cropMap(Est_X_t,lookstep)
        matchedPx, matchedPy, matchedReading, convTotal, coarseConfidence = self.searchToMatch(probSP, estimatedX, estimatedY,
            estimatedTheta, rMeasure, xRangeList, yRangeList, self.searchRadius,
                self.searchHalfRad, coarseSearchStep, estMovingDist, estMovingTheta,fineSearch=False, matchMax=matchMax)        
        #High Resolution
        
    def cropMap(self, GridSize, sigma, missMatchProbAtCoarse):
        ScanRadius = 1.1 * self.OG.max_r + 20
        xRangeList = [self.Est_X_t_1['x'] - ScanRadius, self.Est_X_t_1['x'] + ScanRadius]
        yRangeList = [self.Est_X_t_1['y'] - ScanRadius, self.Est_X_t_1['y'] + ScanRadius]
        X_len, Y_len = int((xRangeList[1] - xRangeList[0]) / GridSize),  int((yRangeList[1] - yRangeList[0]) / GridSize)
        searchSpace = np.log(missMatchProbAtCoarse) * np.ones((Y_len + 1, X_len + 1))
        
        #If beyond the existing OG map then grow the OG map
        self.OG.MapExpansionCheck(xRangeList, yRangeList)
        X_Idx, Y_Idx = self.OG.convertRealXYToMapIdx(xRangeList, yRangeList)
        Map = self.OG.LO_t_i[Y_Idx[0]: Y_Idx[1], X_Idx[0]: X_Idx[1] ]              
        ogMap = Map > 0
        
        ogX = self.OG.Grid_Pos[Y_Idx[0]: Y_Idx[1], X_Idx[0]: X_Idx[1]]
        ogY = self.OG.Grid_Pos[Y_Idx[0]: Y_Idx[1], X_Idx[0]: X_Idx[1]]

        OG_X_OccMap, OG_Y_OccMap = ogX[ogMap], ogY[ogMap]
        ogIdx = self.convertXYToSearchSpaceIdx(OG_X_OccMap, OG_Y_OccMap, xRangeList[0], yRangeList[0], GridSize)
        searchSpace[ogIdx[1], ogIdx[0]] = 0
        probSP = self.generateProbSearchSpace(searchSpace, sigma)
        
        return xRangeList, yRangeList, probSP

    def generateProbSearchSpace(self, searchSpace, sigma):
        probSP = gaussian_filter(searchSpace, sigma=sigma)
        probMin = probSP.min()
        probSP[probSP > 0.5 * probMin] = 0
        return probSP
    
    def convertXYToSearchSpaceIdx(self, px, py, beginX, beginY, unitLength):
        xIdx = int(((px - beginX) / unitLength))
        yIdx = int(((py - beginY) / unitLength))
        return xIdx, yIdx
    
    def MapMatch(self):
        pass
    
if __name__ == '__main__':
    Meas_Z_t_1 =np.array([[1,-1],[1,1]],dtype = np.float64)
    Meas_Z_t = np.array([[1,-1],[0,2]],dtype = np.float64)
    Est_X_t = {'x': 0 , 'y':2 , 'yaw':90 } # np.array([0,0,0],dtype = np.float64)
    Est_X_t_1 ={'x': 0 , 'y':2 , 'yaw':0 } # np.array([0,2,0],dtype = np.float64)
    SM = ICP()   
    R,T,orientation,error = SM.match(Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1)