# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:50:29 2021

@author: MangalDeep
"""
# To resolve VS code error
import sys
from pathlib import Path

from numpy import compat
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
import matplotlib.pyplot as plt
#from  MotionModel import CTRA_Motion_Model

from libs.scan_matching import ICP
from utils.tools import rotate , translate, Lidar_3D_Preprocessing

class Graph():
    def __init__(self):
        self.SM = ICP()
        self.Meas_X_t_1 = np.array([0,0,0])
        self.Meas_Z_t_1 = {}
        Pos = np.zeros((3)) # x,y,yaw,t,id
        self.Node= [{'pos': Pos, 'id':0, 't':0}]
        self.WHlBase = (1.143818969726567 + 1.367382202148434)/2
        constraint = np.zeros((3)) # T = x,y, R = yaw
        self.Edge= [{'con': constraint, 'info_mat':np.eye(3,3), 'node_id':[0,0] ,'error': 0, 'chi2':0}]      
        self.iteration = 0
        
    def create_graph(self,Meas_X_t, Meas_Z_t):
        Meas = np.array([Meas_X_t['x'], Meas_X_t['y'], Meas_X_t['yaw'], Meas_X_t['v'], Meas_X_t['acc'], Meas_X_t['steer'], Meas_X_t['t']])
        Meas_Z_t = Lidar_3D_Preprocessing(Meas_Z_t)
        if self._nodeRequired(Meas):
            self.iteration+=1
            Pose = Meas[0:3]
            self._addNode(Pose, Meas_X_t['t'])
            
            if self.iteration==1:
                self.Meas_Z_t_1 = Meas_Z_t.copy()
                self.Meas_X_t_1 = Meas.copy()
                return None
    
            newNode = self.Node[-1]
            oldNode = self.Node[-2]
            xx1 = self._addOdomEdge(newNode, oldNode, Meas_Z_t) #Z_hat
            print(xx1)
            xx2 = self._addObsvEdge(newNode, oldNode, Meas_Z_t) # Z
            self.Meas_Z_t_1 = Meas_Z_t.copy()
            self.Meas_X_t_1 = Meas.copy()
        
    def _nodeRequired(self,Meas_X_t):
        dist = np.sqrt((self.Meas_X_t_1[0] - Meas_X_t[0])**2 + (self.Meas_X_t_1[1] - Meas_X_t[1])**2)
        yaw = self.Meas_X_t_1[2] - Meas_X_t[2]
        if dist>0.1 or yaw >0.1:
            return True
        else:
            return False
        
    def _addNode(self, newPose,t):
        #State vector
        newNode = {'pos': newPose, 'id':len(self.Node), 't':t}
        self.Node.append(newNode )
    
    def _addOdomEdge(self,newNode, oldNode, Meas_Z_t):
        # Use only if conseqitve measurement - x_i and x_i+1
        #relative pose information
        Xj_from_Xi =self._inverse_pose_composition(newNode, oldNode)
        return Xj_from_Xi
    
    def _addObsvEdge(self, newNode, oldNode, Meas_Z_t):
        '''
        Edge Between Node ids?
        Information Matrix
        Estimate >> from lidar
        Vertices
        '''
        # Use for arbitary measurement - x_i and x_j
        # Construct a virtual measurement using ICP, find the relative transformation
        # xi and xj
        newN = newNode['pos']
        oldN = oldNode['pos']
        Est_X_t_1 = oldN
        Est_X_t = newN
        
        GT, GT_Lst = self.SM.match_LS(Meas_Z_t.to_numpy().T, self.Meas_Z_t_1.to_numpy().T, 
                                                      Est_X_t, 
                                                      Est_X_t_1)
        T = GT['T'].flatten()
        H = self._homogenous_CS(T,GT['r'])
        temp = GT_Lst -Est_X_t_1
        return temp
        # tempEdge = {'con': temp, 'info_mat':np.eye(3,3), 'node_id':[newNode['id'], oldNode['id']]}
        # Er = self._chi2(newNode, oldNode, Meas_Z_t.to_numpy().T, RT)
        # tempEdge.update(Er)
        # self.Edge.append(tempEdge)
    
    def _chi2(self, newNode, oldNode, Meas_Z_t, RT):
        error = self._errorfunc(newNode, oldNode, Meas_Z_t, RT)
        chi2 = error.T @ np.eye(3,3) @ error
        DOF = 3*len(self.Edge) - 3*len(self.Node)
        chi2_red = chi2/DOF
        Er = {'error':error, 'chi2':chi2,'chi2_red':chi2_red}
        return Er
    
    def _errorfunc(self,newNode, oldNode, Meas_Z_t,SM_RT):
        newPose = newNode['pos'].reshape(3,1)
        oldPose = oldNode['pos'].reshape(3,1)
        # Difference between the poses w.r.t graph
        PoseDiff = self._inverse_pose_composition(newPose, oldPose)
        
        # Difference between the poses w.r.t measurement obtained by scan matching
        # Find error function from scan match and pose difference
        error_xy =  rotate(SM_RT['yaw']) @ (rotate(oldPose[2]) @ (newPose[0:2] - oldPose[0:2]) - SM_RT['T'])
        error_yaw = newPose[2,0] - oldPose[2,0] - SM_RT['yaw']
        error_i_j = np.append(error_xy , error_yaw).reshape(3,-1)
        return error_i_j
        
    def _homogenous_CS(self,T,R):
        dim = np.shape(T)[0]
        H = np.zeros((dim+1, dim+1))
        H[dim, dim] = 1.0
        H[0:dim,0:dim] = R
        H[0:dim,dim] = T
        return H
    
    def _inverse_pose_composition(self,Pose1, Pose2):
        diff = (Pose1['pos'][:2] - Pose2['pos'][:2]).reshape(2,1)
        R = rotate(Pose2['pos'][2])
        composed_T = R.T @ diff
        composed_R = Pose1['pos'][2] - Pose2['pos'][2]
        composed_pose = np.append(composed_T, composed_R)
        return composed_pose
    
    def Node_id_pose(self,Idx):
        for N in self.Node:
            if N['id'] ==Idx:
                return N['pos']
            
    def Edge_id_pose(self,Id_i, Id_j):
        for E in self.Edge:
            if np.array_equal(E['id'], [Id_i, Id_j]):
                return E['con']
    def plot(self):
        I = np.zeros(3)
        for N in self.Node:
           I = np.vstack((I, N['pos']))
        plt.plot(I[2:,0], I[2:,1],'g',marker=(6,0, I[1,2]),markersize= 8, label='Graph')
        plt.grid('on')
        plt.legend(loc = 'best')
        plt.show()
        
if __name__ == '__main__':
    Gp = Graph()
    Meas_Z_t = {}
    Meas_X_t = {}
    Gp.create_graph(Meas_X_t , Meas_Z_t )