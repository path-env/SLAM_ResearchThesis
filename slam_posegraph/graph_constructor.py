# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:50:29 2021

@author: MangalDeep
"""

import numpy as np
#from  MotionModel import CTRA_Motion_Model

from libs.scan_matching import ICP
from utils.tools import rotate , translate, Lidar_3D_Preprocessing

class Graph():
    def __init__(self):
        self.SM = ICP()
        self.Meas_X_t_1 = {}
        self.Meas_Z_t_1 = {}
        Pos = np.zeros((3)) # x,y,yaw,t,id
        self.Node= [{'pos': Pos, 'id':0, 't':0}]
        self.WHlBase = (1.143818969726567 + 1.367382202148434)/2
        constraint = np.zeros((3)) # T = x,y, R = yaw
        self.Edge= [{'con': constraint, 'info_mat':np.eye(3,3), 'node_id':[] ,'error': 0, 'chi2':0}]      
        self.iteration = 0
        
    def create_graph(self,Meas_X_t, Meas_Z_t):
        Meas_Z_t = Lidar_3D_Preprocessing(Meas_Z_t)
        self.iteration+=1
        Pose = np.array([*Meas_X_t.values()][3:6])
        self._addNode(Pose, Meas_X_t['t'])
        
        if self.iteration==1:
            self.Meas_Z_t_1 = Meas_Z_t.copy()
            self.Meas_X_t_1 = Meas_X_t.copy()
            return None

        newNode = self.Node[-1]
        oldNode = self.Node[-2]
        self._addObsvEdge(newNode, oldNode, Meas_Z_t)
        self.Meas_Z_t_1 = Meas_Z_t.copy()
        self.Meas_X_t_1 = Meas_X_t.copy()
        
    def _addNode(self, newPose,t):
        #State vector
        newNode = {'pos': newPose, 'id':len(self.Node), 't':t}
        self.Node.append(newNode )
    
    def _addOdomEdge(self,Meas_X_t, IMU_Z_t):
        # Use only if conseqitve measurement - x_i and x_i+1
        # Given a pose the find the new one , MotionModel
        Meas_X_t['yaw_dot'] = -1*IMU_Z_t['ang_vel']
        cmdIn = np.array([Meas_X_t['yaw_dot'], Meas_X_t['acc']]).reshape(2,1)
        #Est_X_t = CTRA_Motion_Model(self.Meas_X_t_1, cmdIn,self.WHlBase, dt=Meas_X_t['t'] - self.Meas_X_t_1['t'])
        #relative pose information
    
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
        new = newNode['pos']
        old = oldNode['pos']
        Est_X_t_1 = {'x': old[0],'y': old[1],'yaw': old[2]}
        Est_X_t = {'x': new[0],'y': new[1],'yaw': new[2]}
        
        GT, RT = self.SM.match(Meas_Z_t.to_numpy().T, self.Meas_Z_t_1.to_numpy().T, 
                                                      Est_X_t, 
                                                      Est_X_t_1)
        T = RT['T'].flatten()
        H = self._homogenous_CS(T,RT['r'])
        temp = np.append(T,RT['yaw']).reshape(3,1)
        tempEdge = {'con': temp, 'info_mat':np.eye(3,3), 'node_id':[newNode['id'], oldNode['id']]}
        Er = self._chi2(newNode, oldNode, Meas_Z_t.to_numpy().T, RT)
        self.Edge.append( tempEdge.update(Er))
    
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
        TransMatrx = np.zeros((3,1))
        TransMatrx[0,0] = (Pose1[0] - Pose2[0]) #*np.cos(np.deg2rad(Pose2['yaw']))
        TransMatrx[1,0] = (Pose1[1] - Pose2[1]) # *np.sin(np.deg2rad(Pose2['yaw']))
        TransMatrx[2,0] =  Pose1[2] - Pose2[2]
        return TransMatrx
    
    def Node_id_pose(self,Idx):
        for N in self.Node:
            if N['id'] ==Idx:
                return N['pos']
            
    def Edge_id_pose(self,Id_i, Id_j):
        for E in self.Edge:
            if np.array_equal(E['id'], [Id_i, Id_j]):
                return E['con']
    def plot(self):
        pass
    
if __name__ == '__main__':
    Gp = Graph()
    Gp.create_graph(Meas_X_t , Meas_Z_t )