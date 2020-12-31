# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 08:51:49 2020

@author: Mangal Deep.B.M
"""

import numpy as np
import symbol as sp
class KalmanFilter_Linear():
    def __init__(self,x , P_Cov,Meas):
        self.P_Cov = [P_Cov]
        self.Q_Cov = np.array([[0.1 , 0],[0,0.1]])
        self.State_vec =  np.array([[0 ,0]]).T
        self.T = sp.Symbol('T')
        self.Mot_Mdl_CV = np.array([[1,self.T],[0,1]])
        self.H = np.array([[1 ,0]])
        self.Meas =Meas;
    
    def Prediction(self):
        #Extrapolate the state
        self.State_pred_vec = self.Mot_Mdl_CV @  self.State_updt_vec + np.array([[0,self.T]]) * 2
        
        # Extrapolate Uncertainity
        self.P_pred_Cov = self.Mot_Mdl_CV @ self.P_updt_Cov @ self.Mot_Mdl_CV.T  + self.Q_Cov
    
    def Update(self):
        #Kalman Gain
        Kalman_G = self.P_pred_Cov @ self.H.T @ np.linalg.inv(self.H @ self.P_pred_Cov @ self.H.T   + 0.1)

        
        # Update Estimate with Meas
        self.State_updt_vec = self.State_pred_vec + Kalman_G @ (self.Meas - self.H@self.State_pred_vec)
    
        # Update Estimate Uncertainity
        self.P_updt_Cov = (np.identity(2) - Kalman_G@self.H) @ self.P_pred_Cov
    
    def mainloop(self):
        for i in range(length(Meas)):
            