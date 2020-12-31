 # -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 09:59:08 2020

@author: MangalDeep
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class ScanMatching_OG():
    #Naming Convention of variables:
        #Z - Measurement
        #k - Time of measurement
        #R_/RR/G - R- / R+ /Global Coordinate Frame
    def __init__(self):
        self.KernelSize = 10

    def CreateNDMap(self,OG):
        #Grab the map constructed so far and construct MAP ND
        self.OG = OG
        self.Map_ND_k_1_G = self.Map_ND()
        
    def Scan_2_Map(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1):
        #ICP to derive R, T, phi
        R ,T_icp,phi_icp ,error =  self.ICP(Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1)
        Rot = np.array([[np.cos(phi_icp),-np.sin(phi_icp)] , [np.sin(phi_icp),np.cos(phi_icp)]]).reshape(2,2)
        Meas = np.array([Meas_Z_t['x'] ,Meas_Z_t['y']]).reshape(2,-1)
        
        Z_k_R_1 = Rot@Meas + T_icp
        
        #In Global Frame
        Meas_Z_R_ = self.CoordinateTransform_t_1(Meas_Z_t,Est_X_t,Est_X_t_1)
        phi = np.deg2rad(Est_X_t_1['yaw'])
        Rot = np.array([[np.cos(phi),-np.sin(phi)] , [np.sin(phi),np.cos(phi)]]).reshape(2,2)
        T = np.array([Est_X_t_1['x'] , Est_X_t_1['y']]).reshape(2,1)
        Meas = np.array([Meas_Z_R_['x'] ,Meas_Z_R_['y']]).reshape(2,-1)
        Z_k_G = Rot@Meas + T
        
        #Robot pose in Global for time K
        Rot = np.array([[np.cos(phi),-np.sin(phi)] , [np.sin(phi),np.cos(phi)]]).reshape(2,2)
        T = np.array([Est_X_t_1['x'] , Est_X_t_1['y']]).reshape(2,1)
        
        Est_X_k_G = Rot @ T_icp + T
        Est_phi_k_G = phi_icp + np.deg2rad(Est_X_t_1['yaw'])
        
        # Technique to correct Est_X_k_G and Est_phi_k_G by using G+
        # Assume where the bot starts is the global index (0,0,0)        
        Meas_Z_G = self.Scan_to_GCS(Meas_Z_t,Est_X_t)

        #Convert New scan to scan ND in global
        Scan_ND_k_G = self.Scan_ND(Meas_Z_t,Est_X_t)
                
        #similiarity between two NDs using KL Divergence to find the matching Map
        Sim_score_MSE = self.Scan_Map_match(Scan_ND_k_G , self.Map_ND_k_1_G)
        
        return Sim_score_MSE
        
    def ICP(self,Meas_Z_t,Meas_Z_t_1,Est_X_t,Est_X_t_1):
        #Transform Coordinates of measurements from  t to t-1
        Meas_Z_R_ = self.CoordinateTransform_t_1(Meas_Z_t,Est_X_t,Est_X_t_1)
        
        #Assume 1 to 1 correspondence
        X_t_mean = np.mean(Meas_Z_t['x'])
        Y_t_mean = np.mean(Meas_Z_t['y'])
        
        XMeas_Z_R__mean = np.mean(Meas_Z_R_['x'])
        YMeas_Z_R__mean = np.mean(Meas_Z_R_['y'])
        
        #MeanCentered
        X_t_MC = Meas_Z_t['x'] - X_t_mean
        Y_t_MC = Meas_Z_t['y'] - Y_t_mean
        
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
                 
    def CoordinateTransform_t_1(self,Meas_Z_t,Est_X_t,Est_X_t_1):
        Meas_Z_R_ = {}
        X = Meas_Z_t['x']
        Y = Meas_Z_t['y']
        Z = np.array([X,Y]).reshape(2,-1)
        dx = Est_X_t['x'] - Est_X_t_1['x']
        dy = Est_X_t['y'] - Est_X_t_1['y']
        T = np.array([dx,dy]).reshape(2,1)
        dyaw =np.deg2rad(Est_X_t['yaw'] - Est_X_t_1['yaw'])
        R = np.array([[np.cos(dyaw),-np.sin(dyaw)] , [np.sin(dyaw),np.cos(dyaw)]]).reshape(2,2)
        Meas_t = R@Z + T
        Meas_Z_R_['x'] = Meas_t[0,:]
        Meas_Z_R_['y'] = Meas_t[1,:]
        Meas_Z_R_['Azimuth'] =np.rad2deg(np.arctan2(Meas_Z_R_['y'] , Meas_Z_R_['x']))
        return Meas_Z_R_
    
    def Scan_to_GCS(self,Meas_Z_t,Est_X_t,Pose_X_t = {'x':0 , 'y':0 , 'yaw':0}):
        Meas_Z_G = {}
        X = Meas_Z_t['x']
        Y = Meas_Z_t['y']
        Z = np.array([X,Y]).reshape(2,-1)
        dx = Est_X_t['x'] - Pose_X_t['x']
        dy = Est_X_t['y'] - Pose_X_t['y']
        T = np.array([dx,dy]).reshape(2,1)
        dyaw =np.deg2rad(Est_X_t['yaw'] - Pose_X_t['yaw'])
        R = np.array([[np.cos(dyaw),-np.sin(dyaw)] , [np.sin(dyaw),np.cos(dyaw)]]).reshape(2,2)
        Meas_t = R@Z + T
        Meas_Z_G['x'] = Meas_t[0,:]
        Meas_Z_G['y'] = Meas_t[1,:]
        Meas_Z_G['Azimuth'] =np.rad2deg(np.arctan2(Meas_Z_G['y'] , Meas_Z_G['x']))
        return Meas_Z_G

    def Map_to_GCS(self,Map,Pose_X_t = {'x':0 , 'y':0 , 'yaw':0}):
        #Only coordinates are processed not the entire matrix
        Map_G = [0,1]
        Global_CG = self.OG.getMapPivotPoint()        
        Map_G[0] =    Global_CG[0] - Map[0]
        Map_G[1] =    Global_CG[1] - Map[1]
        return Map_G[0] , Map_G[1]    
    
    def Scan_ND(self,Meas_Z_t,Pose_X_t):
        #Parameter fopr rounding the coordinates
        ### Create ND  Grid cells
        #Meas_Z_t = Meas_Z_t.assign(X_round =Meas_Z_t.round({'x':0})['x'])
        #Meas_Z_t = Meas_Z_t.assign(Y_round =Meas_Z_t.round({'y':0})['y'])
        #SimCoord = Meas_Z_t.groupby([Meas_Z_t['X_round'] , Meas_Z_t['Y_round']]).size()
        #get occupancy info
        #Meas_Z_G = np.array([Meas_Z_G['x'] , Meas_Z_G['y']]).reshape(2,-1)
        #Scan_ND_k_G = self.getOccupancyInfo(Meas_Z_G)
        ScanMap = self.OG.getScanMap(Meas_Z_t,Pose_X_t)
        Scan_ND_k_G,Scan_ND_Map = self.getOccupancyInfo2(ScanMap)
        
        return Scan_ND_k_G
        '''
        # #Different cells in a map
        # for N in range(len(SimCoord)):
        #     #Consider  a cell only with 2 or more detections
        #     if SimCoord.values[N] >2:
        #         (x,y) = SimCoord.index[N]
        #         rows = np.where(np.logical_and(Meas_Z_t['X_round'] == x , Meas_Z_t['Y_round'] == y))
        #         #ND- Mean
        #         x_mean  = np.sum(Meas_Z_t.loc[rows,'x'])/len(rows)
        #         y_mean  = np.sum(Meas_Z_t.loc[rows,'y'])/len(rows)
        #         Scan_Z_k_G['x_mean'].append(x_mean)
        #         Scan_Z_k_G['y_mean'].append(y_mean)
        #         Scan_Z_k_G['j'].append(N)
        #         Scan_Z_k_G['cell_XIndx'].append(x)
        #         Scan_Z_k_G['cell_YIndx'].append(y)
        #         Scan_Z_k_G['NOD'].append(SimCoord.values[N])
                
        #         #ND Covariance
        #         MC_x = np.sum(Meas_Z_t.loc[rows,'x']) - x_mean
        #         MC_y = np.sum(Meas_Z_t.loc[rows,'y']) - y_mean
        #         MC = np.array([MC_x , MC_y]).reshape(2,-1)
        #         Tmp = np.sum(np.matmul(MC, MC.T)) /len(rows)                
        #         Scan_Eps_k_G['j'].append(N)
        #         Scan_Eps_k_G['cov'].append(Tmp) 
        '''
         
    def getOccupancyInfo(self,Mat):
        CellInfo = []
        self.ND_scan = np.zeros(self.OG.Local_Map.shape)
        j = 0 # cell number
        for row in range(self.OG.Xlim_start ,self.OG.Xlim_end,self.KernelSize):
            row_idx = np.where(np.logical_and(Mat[0]>=row,Mat[0]<row+self.KernelSize))
            row_idx = row_idx[0]
            for col in range(self.OG.Ylim_start ,self.OG.Ylim_end,self.KernelSize):
                #self.ND_scan[row-OG.Xlim_start:row+self.KernelSize , col-OG.Ylim_start:col+self.KernelSize] = 
                col_idx = np.where(np.logical_and(Mat[1]>=col,Mat[1]<col+self.KernelSize))
                col_idx = col_idx[0]
                #Check if it is captured in row_idx
                ids = np.intersect1d(row_idx,col_idx)
                Tmp = {} 
                if len(ids):
                    #ND- Mean
                    Tmp['x_mean'] = np.sum(Mat[0,ids])/len(ids)
                    Tmp['y_mean'] = np.sum(Mat[1,ids])/len(ids)
                    Tmp['j'] = j
                    Tmp['cell_XIndx'] = Mat[0,ids]
                    Tmp['cell_YIndx'] = Mat[1,ids]
                    Tmp['NOD'] = len(ids)                    
                    #ND - Covariance
                    MC_x = Mat[0,ids] - Tmp['x_mean']
                    MC_y = Mat[1,ids] - Tmp['y_mean']
                    MC = np.array([MC_x , MC_y]).reshape(2,-1)
                    Tmp['cov'] = (np.matmul(MC, MC.T)) /len(ids)       
                    CellInfo.append(Tmp)
                else:
                    Tmp['j'] = j
                    CellInfo.append(Tmp)
                j+=1
        return CellInfo
        
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
                jj +=1
        return ND,ND_cells       
    
    def Map_ND(self):        
        OccVal = self.OG.l_occ
        Map = np.rot90(self.OG.Local_Map)
        # Map_Ray_corres = self.OG.Map2Ray_corres        
        # # Convert correspondences to global
        # for i in range(len(Map_Ray_corres)):
        #     Coord = np.array(Map_Ray_corres[i]['x_coord'],Map_Ray_corres[i]['y_coord'] )
        #     Map_Ray_corres[i]['x_coord'] , Map_Ray_corres[i]['y_coord'] = self.Map_to_GCS()        
        KernelSize = self.KernelSize
        #Appply on Map recursively creating cells j
        j =0
        Map_ND_k_1_G =[]
        self.ND_map = np.zeros(Map.shape)
        for row in range(0,self.OG.Long_Length, KernelSize):
            for col in range(0,self.OG.Lat_Width , KernelSize):       
                Tmp = {}
                Tmp['j'] = j
                map_extract = Map[row:KernelSize+row,col:KernelSize+col]
                ND_kernel= gaussian_filter(map_extract, sigma = 2) ## Only a representation
                
                if np.any(map_extract >=OccVal*2):
                    self.ND_map[row:KernelSize+row,col:KernelSize+col] = map_extract
                    #ND -Mean
                    #ff = np.argmin(np.abs(np.subtract(ND_kernel,np.mean(ND_kernel))))
                    #mn = ND_kernel.flatten()[ff]
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
                    Map_ND_k_1_G.append(Tmp)                               
                j +=1
        '''
        #As per text 3.4
        for jj in self.Map_ND_k_1_G:
                    Tmp['cell_XIndx'] = (occ_coord[0])
                    Tmp['cell_YIndx'] = (occ_coord[1])
                    Tmp['NOD'] = np.round(np.max(map_extract)/OccVal)
                    Tmp['x_mean'] = (Tmp['NOD']*Tmp['x_mean'] + Scan_ND['NOD']*Scan_ND['x_mean'] )/ (Tmp['NOD'] + Scan_ND['NOD'])
                    Tmp['y_mean'] = (Tmp['NOD']*Tmp['y_mean'] + Scan_ND['NOD']*Scan_ND['y_mean'] )/ (Tmp['NOD'] + Scan_ND['NOD'])
                    
                    #ND Covariance
                    Tmp['cov'] = (Tmp['NOD']*Tmp['cov'] + Scan_ND['NOD']*Scan_ND['cov'] )/ (Tmp['NOD'] + Scan_ND['NOD'])
                Map_ND_k_1_G.append(Tmp)                               
                j +=1
        
        
        '''
        self.PlotMapND()
        return Map_ND_k_1_G

        
    def Scan_Map_match(self,Scan_ND , Map_ND):
        ND_Dim = 2
        Map_N_j = len(Map_ND)
        Scan_N_j = len(Scan_ND)
        if Map_N_j != Scan_N_j:
            return False
        self.KL = []
        for j in range(Map_N_j):
            if 'x_mean' in Map_ND[j].keys() and 'x_mean' in Scan_ND[j].keys():
                Diff_x = Map_ND[j]['x_mean']- Scan_ND[j]['x_mean']
                Diff_y = Map_ND[j]['y_mean'] - Scan_ND[j]['y_mean']                       
                #KL Divergence
                P2_2 = np.linalg.inv(Map_ND[j]['cov']) 
                P1 = np.trace( np.matmul(P2_2 ,Scan_ND[j]['cov'] ))
                P2_1 = np.array([Diff_x , Diff_y ])
                P2_3 = P2_1.reshape(2,1)
                P3 = np.log( np.linalg.det(Scan_ND[j]['cov'] ) / np.linalg.det(Map_ND[j]['cov'] ))
                dist = -0.5* (P1 +  P2_1@P2_2@P2_3 - P3 -ND_Dim)[0]
            elif 'x_mean' in Map_ND[j].keys() and  'x_mean' not in Scan_ND[j].keys():
                dist = 0#'NA in Scan'
            elif 'x_mean' in Scan_ND[j].keys() and 'x_mean' not in Map_ND[j].keys():
                dist =0# 'NA in Map'
            else:   
                dist = 0#'NA' 
            self.KL.append(dist)
        ##MSE
        #Error - Dist
        #Equation seems to do matrix square, hence skip
        # TO find only the mean value
        SE = 0
        sss = [x for x in self.KL]
        MSE = np.sum(np.array(sss))
        # for jj in self.KL:
        #     if jj == 492:
        #         print(jj)
        #     if jj == "NA in Map":
        #         pass #return Warning   #Check the global Map
        #     elif jj == 'NA in Scan':
        #         pass #return Warning   #Check the scan
        #     if type(jj) == int:
        #         SE =SE+ jj
        # MSE = SE/ len(self.KL)   
        return MSE

    def PlotMapND(self):
        probMap = np.exp(self.ND_map)/(1.+np.exp(self.ND_map)) 
        plt.title("Gaussian Filtered Plot of Map")
        # plt.xlim(self.Ylim_start , self.Ylim_end)
        # plt.ylim(self.Xlim_start , self.Xlim_end)                
        plt.imshow(probMap, cmap='Greys')
        plt.draw()
        plt.show() 
        
        
    def EstimatePose(self):
        pass
    
    def CrctGCS(self):
        pass
    
    def CorrectPose(self):
        pass        