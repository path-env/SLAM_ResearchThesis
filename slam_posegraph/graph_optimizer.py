# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:31:27 2021

@author: MangalDeep
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from slam_posegraph.graph_constructor import Graph
from utils.tools import rotate , translate

class ManifoldOptimizer():
    def __init__(self):
        self.conv_thresh = 1*10^-3

        
    def optimize(self,Graph,iterations = 20):
        self.Gph = Graph
        Nodes = self.Gph.Node
        self.A = np.zeros((len(Nodes), len(Nodes)))
        self.B = np.zeros((len(Nodes), 0))
        self.H = np.zeros((len(Nodes), len(Nodes)))
        #self._construct_Matrx()
        cnt = 0
        while cnt < 20 :
            for E in self.Gph.Edge:
                # Compute Jacobian
                id_i, id_j = E['node_id']
                N_Pos = [self.Gph.Node_id_pose(id_i), self.Gph.Node_id_pose(id_j)]
                A,B = self._calc_Jacobian(N_Pos, E)
                # Operate on Manifold\
                # In 2D Manifolds are identity matrix
                A_MF = A @ np.eye(3,3)
                B_MF = B @ np.eye(3,3)
                
                #Compute Hessian
                H = self._calc_Hessian(E, A_MF, B_MF)
                
                #Compute coefficient vector
                B = self._calc_CoEff(E,A_MF, B_MF)
                
            # Choldesky factorization
            del_x = spsolve(H,-B)
            L = np.linalg.cholesky(H)
            
            del_x = -1* np.linalg.inv(L) @ B
            
            for N in self.Gph.Node:
                self.Gph.Node[N]['pos'] += del_x
            
            cnt += 1
            
        # for Edg in self.Gph.Error:
        #     H = self._calc_Hessian(A, B)
        
    def _construct_Matrx(self):
        for E in self.Gph.Edge:
            # Construct the Jacobian matrix
            id_i, id_j = E['node_id']
            N_Pos = [self.Gph.Node_id_pose(id_i), self.Gph.Node_id_pose(id_j)]
            self._calc_Jacobian(N_Pos, E)
            self.info_mtrx[id_i:id_j, id_i:id_j] = E['info_mat']
        self._calc_Hessian()
        self._calc_CoEff()
        
    def _calc_Jacobian(self,N_Pos, E):
        i_pos, j_pos = N_Pos[0], N_Pos[1]
        t_i, yaw_i = i_pos[:2],np.deg2rad(i_pos[2])
        t_j, yaw_j = j_pos[:2],np.deg2rad(j_pos[2])
        id_i, id_j = E['node_id']
        E_Con = E['con']
        d_rotate = np.array([-np.sin(yaw_i), np.cos(yaw_i), -np.cos(yaw_i), -np.sin(yaw_i)]).reshape(2,2)
        
        A = np.zeros((3,3))
        A[0:2,0:2] += -rotate(E_Con[2]).T @ rotate(yaw_i)
        A[:,0:2] += rotate(E_Con[2]).T @ d_rotate @ (t_j - t_i)
        A[2,2] += -1
        
        B = np.zeros((3,3))
        B[0:2,0:2] += rotate(E_Con[2]).T @ rotate(yaw_i)
        B[2,2] += 1
        return A,B
    
    def _calc_Hessian(self,E,A_MF, B_MF):
        H = np.zeros((6,6))
        H[0:3,0:3] += A_MF.T @ E['info_mat'] @ A_MF
        H[0:3,3:6] += A_MF.T @ E['info_mat'] @ B_MF
        H[3:6,0:3] += B_MF.T @ E['info_mat'] @ A_MF
        H[3:6,3:6] += B_MF.T @ E['info_mat'] @ B_MF        
        return H
    
    def _calc_CoEff(self,E,A_MF, B_MF):
        B = np.array((2,1))
        B[0] += A_MF.T @ E['info_mat'] * E['error']
        B[1] += B_MF.T @ E['info_mat'] * E['error']
        return B
        
class ManifoldOptimizerWIp():
    def __init__(self):
        self.conv_thresh = 1*10^-3

        
    def optimize(self,Graph,iterations = 20):
        self.Gph = Graph
        Nodes = self.Gph.Node
        self.A = np.zeros((len(Nodes), len(Nodes)))
        self.B = np.zeros((len(Nodes), 0))
        self.H = np.zeros((len(Nodes), len(Nodes)))
        #self._construct_Matrx()
        while iterations < 20 :
            for E in self.Gph.Edge:
                # Compute Jacobian
                id_i, id_j = E['node_id']
                N_Pos = [self.Gph.Node_id_pose(id_i), self.Gph.Node_id_pose(id_j)]
                A,B = self._calc_Jacobian(N_Pos, E)
                # Operate on Manifold\
                # In 2D Manifolds are identity matrix
                A_MF = self.A @ np.eye(2,2)
                B_MF = self.B @ np.eye(2,2)
                
                #Compute Hessian
                H = self._calc_Hessian(A_MF, B_MF)
                
                #Compute coefficient vector
                B = self._calc_CoEff(A_MF, B_MF)
                
            # Choldesky factorization
            del_x = spsolve(H,-B)
            L = np.linalg.cholesky(H)
            
            del_x = -1* np.linalg.inv(L) @ B
            
            for N in self.Gph.Node:
                self.Gph.Node[N]['pos'] += del_x
            
            iterations += 1
            
        # for Edg in self.Gph.Error:
        #     H = self._calc_Hessian(A, B)
        
    def _construct_Matrx(self):
        for E in self.Gph.Edge:
            # Construct the Jacobian matrix
            id_i, id_j = E['node_id']
            N_Pos = [self.Gph.Node_id_pose(id_i), self.Gph.Node_id_pose(id_j)]
            self._calc_Jacobian(N_Pos, E)
            self.info_mtrx[id_i:id_j, id_i:id_j] = E['info_mat']
        self._calc_Hessian()
        self._calc_CoEff()
        
    def _calc_Jacobian(self,N_Pos, E):
        i_pos, j_pos = N_Pos[0], N_Pos[1]
        t_i, yaw_i = i_pos[:2],np.deg2rad(i_pos[2])
        t_j, yaw_j = j_pos[:2],np.deg2rad(j_pos[2])
        id_i, id_j = E['node_id']
        E_Con = E['con']
        d_rotate = np.array([-np.sin(yaw_i), np.cos(yaw_i), -np.cos(yaw_i), -np.sin(yaw_i)]).reshape(2,2)
        A = np.zeros((3,3))
        A[0:2,0:2] += -rotate(E_Con[2]).T @ rotate(yaw_i)
        A[:,0:2] += rotate(E_Con[2]).T @ d_rotate @ (t_j - t_i)
        A[2,2] += -1
        
        B += rotate(E_Con[2]).T @ rotate(yaw_i)
        B += 0
    
    def _calc_Hessian(self,A_MF, B_MF):
        self.H = self.A.T @ self.info_mtrx @ self.B
        self.H[0,0] = 1
    
    def _calc_CoEff(self,A_MF, B_MF):
        B = np.array((2,1))
        B[0] += A_MF.T @ self.Error[-1]['info_mat'] @ self.Error[-1]['error']
        B[1] += B_MF.T @ self.Error[-1]['info_mat'] @ self.Error[-1]['error']