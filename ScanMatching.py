# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 09:59:08 2020

@author: MangalDeep
"""
import numpy as np

class ScanMatching_OG():
    def __init__(self):
        pass
    
    def ICP(self,Meas_Z_t,Meas_X_t,Meas_Z_t_1,Meas_X_t_1):
        #ND
        self.ND_grids( Meas_Z_t_1 , Meas_Z_t)
        #Transform Coordinates of measurements from  t to t-1
        trans_Z_t = self.CoordinateTransform_t_1(Meas_Z_t,Meas_X_t,Meas_X_t_1)
        
        #Assume 1 to 1 correspondence
        X_t_mean = np.mean(Meas_Z_t['x'])
        Y_t_mean = np.mean(Meas_Z_t['y'])
        
        Xtrans_Z_t_mean = np.mean(trans_Z_t['x'])
        Ytrans_Z_t_mean = np.mean(trans_Z_t['y'])
        
        #MeanCentered
        X_t_MC = Meas_Z_t['x'] - X_t_mean
        Y_t_MC = Meas_Z_t['y'] - Y_t_mean
        
        Xtrans_Z_t_MC = trans_Z_t['x'] - Xtrans_Z_t_mean
        Ytrans_Z_t_MC = trans_Z_t['y'] - Ytrans_Z_t_mean
        
        #SVD Method
        q_t_1 = np.array([Xtrans_Z_t_MC, Ytrans_Z_t_MC]).reshape(2,-1)
        q_t = np.array([X_t_MC, Y_t_MC]).reshape(2,-1)
        H = np.matmul(q_t_1 ,q_t.T )
        u, s, vh = np.linalg.svd(H,full_matrices=True,compute_uv=True)
        
        #Find Roation
        R = np.matmul(u,vh)
        if np.linalg.det(R) <= -1:
            Warning('No Unique solution obtained')
        T = np.array([Xtrans_Z_t_mean , Ytrans_Z_t_mean]).reshape(2,1) - R@np.array([X_t_mean , Y_t_mean]).reshape(2,1)          
        error = np.sum(np.hypot(Xtrans_Z_t_MC , Ytrans_Z_t_MC)**2) + np.sum(np.hypot(X_t_MC , Y_t_MC)**2) - 2*np.sum(s)
        orientation = np.arctan2(R[1,0],R[0,0])
        return R,T,error,orientation
                 
    def CoordinateTransform_t_1(self,Meas_Z_t,Meas_X_t,Meas_X_t_1):
        trans_Z_t = {}
        X = Meas_Z_t['x']
        Y = Meas_Z_t['y']
        Z = np.array([X,Y]).reshape(2,-1)
        dx = Meas_X_t_1['x'] - Meas_X_t['x']
        dy = Meas_X_t_1['y'] - Meas_X_t['y']
        T = np.array([dx,dy]).reshape(2,1)
        dyaw =np.deg2rad(Meas_X_t_1['yaw'] - Meas_X_t['yaw'])
        R = np.array([[np.cos(dyaw),-np.sin(dyaw)] , [np.sin(dyaw),np.cos(dyaw)]]).reshape(2,2)
        Meas_t = R@Z + T
        trans_Z_t['x'] = Meas_t[0,:]
        trans_Z_t['y'] = Meas_t[1,:]
        trans_Z_t['Range_XY_plane'] = np.hypot(Meas_Z_t['x'],Meas_Z_t['y']).to_numpy()
        return trans_Z_t
    
    def CoordTrans_t_1_to_GCS(self):
        pass    
    
    def EstimatePose(self):
        pass
    
    def CrctGCS(self):
        pass
    
    def CorrectPose(self):
        pass
    
    def ND_grids(self,Meas_Z_t_1 , Meas_Z_t):
        #Parameter fopr rounding the coordinates
        precision = 0
        ### Create ND  Grid cells
        Meas_Z_t_1 = Meas_Z_t_1.assign(X_round =Meas_Z_t_1.round({'x':0})['x'])
        Meas_Z_t_1 = Meas_Z_t_1.assign(Y_round =Meas_Z_t_1.round({'y':0})['y'])
        SimCoord = Meas_Z_t_1.groupby([Meas_Z_t_1['X_round'] , Meas_Z_t_1['Y_round']]).size()
        #Initialize cells
        Z_k_1 = []
        Eps_k_1 = []
        for N in range(len(SimCoord)):
            if SimCoord.values[N] >2:
                (x,y) = SimCoord.index[N]
                rows = np.where(np.logical_and(Meas_Z_t_1['X_round'] == x , Meas_Z_t_1['Y_round'] == y))
                #ND- Mean
                x_mean  = np.sum(Meas_Z_t_1.loc[rows,'x'])/len(rows)
                y_mean  = np.sum(Meas_Z_t_1.loc[rows,'y'])/len(rows)
                Z_k_1.append(  np.array(x_mean , y_mean.reshape(2,1)))
                
                #ND Covariance
                MC_x = np.sum(Meas_Z_t_1.loc[rows,'x']) - x_mean
                MC_y = np.sum(Meas_Z_t_1.loc[rows,'y']) - y_mean
                MC = np.array([MC_x , MC_y]).reshape(2,-1)
                Tmp = np.sum(np.matmul(MC, MC.T)) /len(rows)                
                Eps_k_1.append( Tmp )
                
        ### Check if Meas_Z_t sees those cells
        Meas_Z_t = Meas_Z_t.assign(X_round =Meas_Z_t.round({'x':0})['x'])
        Meas_Z_t = Meas_Z_t.assign(Y_round =Meas_Z_t.round({'y':0})['y'])
        SimCoord = Meas_Z_t_1.groupby([Meas_Z_t['X_round'] , Meas_Z_t['Y_round']]).size()