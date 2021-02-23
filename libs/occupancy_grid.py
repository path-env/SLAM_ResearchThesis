# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 09:14:05 2020

@author: Mangal
"""
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging

from libs.remove_ground_plane import RANSAC,Zbased
from utils.tools import Lidar_3D_Preprocessing
plt.ion()
class Map():
    def __init__(self, Poses=None ,MapMode = 1,sceneName = None):
        fig,self.ax = plt.subplots()
        self.breath = 1.2
        self.FOV = 5
        self.max_lidar_r = 50 # For carla
        self.mapSize =  2*self.max_lidar_r + 91
        self.l_occ = np.log(0.65/0.35)
        self.l_free = np.log(0.35/0.65)
        self.l_unknown = 0.5
        self.MapMode = 2
        self.Roffset = 5
        self.SceneName = sceneName
        self.Grid_resol = 1 # a x a m cell
        self.Angular_resol = np.rad2deg(np.arctan( self.Grid_resol/ self.max_lidar_r))
        
        if self.MapMode ==1: #Local Map
            self.Xlim_start = 0
            self.Xlim_end = (2*self.max_lidar_r + 91)
            self.Ylim_start = (0)
            self.Ylim_end = (2*self.max_lidar_r + 91)
            self.Lat_Width =  (self.Ylim_end -  self.Ylim_start)
            self.Long_Length =  (self.Xlim_end -  self.Xlim_start)
        else: #Global Map
            self.Grid_resol = 1
            self.Xlim_start = (0- self.max_lidar_r - 95)
            self.Xlim_end = (0 + self.max_lidar_r + 96)
            self.Ylim_start = (0 - self.max_lidar_r - 95)
            self.Ylim_end = (0 + self.max_lidar_r + 96)    
            self.Lat_Width = np.int32((self.Ylim_end -  self.Ylim_start)/self.Grid_resol)
            self.Long_Length = np.int32((self.Xlim_end -  self.Xlim_start)/self.Grid_resol)
            
        self._createGrid()
        self.Local_Map = np.zeros(())
        cc = ( 0.5 * np.ones((self.Lat_Width,self.Long_Length)))
        self.LO_t = np.log(np.divide(cc,np.subtract(1,cc)))
        self.LO_t_i = self.LO_t   
        self.logger = logging.getLogger('ROS_Decode.OG_MAP')
        self.logger.info('OG_MAP Initialized')
        
    def Update(self,Pose_X_t,Meas_Z_t,PltEnable):
        '''
        Parameters
        ----------
        Meas_Z_t : Dict with keys X, Y, Z , Ring Indices
        Pose_X_t : tuple X,Y, theta(degrees)
            Input in cartisean coordinate system.
        Returns
        -------
        LO_t_i : numpy array
            DESCRIPTION:
            Updated GLobal Map
        '''
        if 'Range_XY_plane' not in Meas_Z_t.keys():
            Meas_Z_t = self.Lidar_3D_Preprocessing(Meas_Z_t)
        InvModel = self.__inverse_sensor_model2(Pose_X_t, Meas_Z_t,PltEnable)
        #self.LO_t_i = np.log(np.divide(InvModel,np.subtract(1,InvModel)))  + self.LO_t_i  - self.LO_t
        self.LO_t = np.zeros(( self.Lat_Width,self.Long_Length))
        self.LO_t_i = InvModel + self.LO_t_i  - self.LO_t
        #Plotting
        if PltEnable == True:
            # self.PlotMap(self.LO_t_i, Pose_X_t, 'GlobalMap')
            pass
        #return self.LO_t_i
     
    def __inverse_sensor_model(self,Pose_X_t, Meas_Z_t,PltEnable = False):
        if self.MapMode ==1:
            (x,y,orientation) = (Pose_X_t[0] ,Pose_X_t[1] ,Pose_X_t[2])
        else:
            (x,y,orientation) = (Pose_X_t[0] ,Pose_X_t[1] ,Pose_X_t[2])
        #Check if expansion required and expand
        self.MapExpansionCheck(x,y)
        self.Local_Map = np.zeros((self.Lat_Width,self.Long_Length))
        Range_vec = Meas_Z_t['Range_XY_plane'].values
        Azi_vec =(np.array(Meas_Z_t['Azimuth'])) - orientation        
        Limits = np.int32(self.mapSize/2)
        
        for row_ind in range(-Limits, Limits):
            for col_ind in range(-Limits,Limits):
                R2cell = np.hypot((row_ind - x),(col_ind - y))
                Azi2cell = np.degrees(np.arctan2((col_ind - y),(row_ind-x))) -orientation
                Almostthere = np.argmin(np.abs(np.subtract(Azi2cell,Azi_vec)))   
                Range_meas = Range_vec[Almostthere]
                Azi_meas = Azi_vec[Almostthere]
                if R2cell > min(self.max_lidar_r,(Range_meas +(self.breath/2))) or (abs(Azi2cell-Azi_meas) >self.FOV/2):
                    self.Local_Map[row_ind- (-Limits), col_ind-(-Limits)] = self.l_unknown
                    continue
                if Range_meas < self.max_lidar_r and  (abs(R2cell - Range_meas)<(self.breath/2)):
                    self.Local_Map[row_ind- (-Limits), col_ind-(-Limits)] =  self.l_occ
                    continue
                if R2cell < Range_meas:
                    self.Local_Map[row_ind- (-Limits), col_ind-(-Limits)] = self.l_free
        #Plotting
        if PltEnable == True:
           self.PlotMap(self.Local_Map, Pose_X_t,'LocalMap')
           pass
        #self.Local_Map[Limits, Limits] = 2   #Location of the Lidar
        return self.Local_Map
        
    def __inverse_sensor_model2(self,Pose_X_t, Meas_Z_t,PltEnable = False):
        if self.MapMode ==1:
            (x,y,orientation) = (Pose_X_t[0] ,Pose_X_t[1]  ,Pose_X_t[2])
        else:
            (x,y,orientation) = (Pose_X_t[0] ,Pose_X_t[1] ,Pose_X_t[2])
        #Check if expansion required and expand
        self.MapExpansionCheck(x,y)
        self.Local_Map = np.zeros((self.Lat_Width,self.Long_Length))
        dx = self.Grid_Pos.copy()
        dx[0, :, :] = np.float16(dx[0, :, :]) - x# A matrix of all the x coordinates of the cell
        dx[1, :, :] = np.float16(dx[1, :, :]) - y# A matrix of all the y coordinates of the cell
        theta_grid = np.array(np.rad2deg(np.arctan2(dx[1, :, :], dx[0, :, :])) - (orientation ))

        # Wrap to +pi / - pi
        theta_grid[theta_grid > 180] -= 360
        theta_grid[theta_grid < -180] += 360

        self.dist_grid = scipy.linalg.norm(dx, axis=0) # matrix of L2 distance to all cells from robot
        for i in Meas_Z_t.iterrows():
            rng = i[1]['Range_XY_plane'] # range measured
            azi = i[1]['Azimuth'] # bearing measured
            free_mask = (np.abs(theta_grid - azi) <= self.FOV/2.0) & (self.dist_grid < (rng - self.breath/2.0))
            occ_mask = (np.abs(theta_grid - azi) <= self.FOV/2.0) & (np.abs(self.dist_grid - rng) <= self.breath/2.0)

            # Adjust the cells appropriately
            self.Local_Map[occ_mask] += self.l_occ
            self.Local_Map[free_mask] += self.l_free

        self.Local_Map = np.fliplr(self.Local_Map)
        self.Local_Map = np.flipud(self.Local_Map)
        #Plotting
        if PltEnable == True:
           self.PlotMap(self.Local_Map, Pose_X_t,'LocalMap')
           pass
        return self.Local_Map
    
    def PlotMap(self,Map,Pose_X_t,title):
        Veh = patches.Rectangle((Pose_X_t[0] -3.5 , Pose_X_t[1]-5),5,0.3,linewidth=1,edgecolor='r')          
        probMap = np.exp(Map)/(1.+np.exp(Map)) 
        plt.title(f"{title} x:{np.round(Pose_X_t[0],5)} , y:{np.round(Pose_X_t[1],5)}, yaw:{np.round(Pose_X_t[2],5)}")
        # plt.ylim(self.Ylim_start , self.Ylim_end)
        # plt.xlim(self.Xlim_start , self.Xlim_end)
        self.ax.add_patch(Veh)                
        plt.imshow(probMap, cmap='Greys')
        #plt.savefig('/Local/Local{title}.png')        
        plt.pause(0.001)
    
    def MapExpansionCheck(self, x, y):
        X_Lim = np.array([np.round(x-(self.max_lidar_r+self.Roffset)) , np.round(x+self.max_lidar_r+self.Roffset)])
        Y_Lim = np.array([np.round(y-(self.max_lidar_r+self.Roffset)) , np.round(y+self.max_lidar_r+self.Roffset)])
        quadrant =  self.ExpansionDirection(X_Lim, Y_Lim)
        while (quadrant != -1):
            self.logger.info('Exapnding the Map in quadrant=%d',quadrant)
            #self.expandOccupancyGrid(quadrant)
            quadrant = self.ExpansionDirection(X_Lim, Y_Lim)

    def ExpansionDirection(self,X_Lim, Y_Lim):
        if any(X_Lim < self.Xlim_start):
            quadrant =1
            self.ExpandQuadrant(1,X_Lim, Y_Lim) #expand down
        elif any(X_Lim > self.Xlim_end):
            quadrant =2
            self.ExpandQuadrant(2, X_Lim, Y_Lim) #expand up
        elif any(Y_Lim < self.Ylim_start):
            quadrant =3
            self.ExpandQuadrant(3, X_Lim, Y_Lim) #expand left
        elif any(Y_Lim > self.Ylim_end):
            quadrant =4
            self.ExpandQuadrant(4, X_Lim, Y_Lim) #expand right
        else:
            self._createGrid()
            quadrant = -1
        return quadrant
   
    def ExpandQuadrant(self, Expansionmode, X_Lim, Y_Lim):
        GridShape = self.Grid_Pos.shape
        Roffset = 20
        if Expansionmode ==1:
            pad_w = np.abs(X_Lim[0] - self.Xlim_start)
            pad_w = pad_w.astype(np.int32)
            self.LO_t_i = np.pad(self.LO_t_i, ((0,0),(0,pad_w)), 'constant', constant_values=(0,0))
            self.Xlim_start = X_Lim[0]
        if Expansionmode ==2:
            pad_w = np.abs(X_Lim[1] - self.Xlim_end)
            pad_w = pad_w.astype(np.int32)
            self.LO_t_i = np.pad(self.LO_t_i, ((0,0),(pad_w,0)), 'constant', constant_values=(0,0))
            self.Xlim_end = X_Lim[1]
        if Expansionmode ==3:
            pad_w = np.abs(Y_Lim[0] - self.Ylim_start)
            pad_w = pad_w.astype(np.int32)
            self.LO_t_i = np.pad(self.LO_t_i, ((0,pad_w),(0,0)), 'constant', constant_values=(0,0))
            self.Ylim_start = Y_Lim[0]
        if Expansionmode ==4:
            pad_w = np.abs(Y_Lim[1] - self.Ylim_end)
            pad_w = pad_w.astype(np.int32)
            self.LO_t_i = np.pad(self.LO_t_i, ((pad_w,0),(0,0)), 'constant', constant_values=(0,0))
            self.Ylim_end = Y_Lim[1]

    def _createGrid(self):
        MGrid = np.meshgrid( np.arange(self.Xlim_start,self.Xlim_end,self.Grid_resol), 
                            np.arange(self.Ylim_start,self.Ylim_end,self.Grid_resol))
        self.Lat_Width = np.int32((self.Ylim_end -  self.Ylim_start)/self.Grid_resol)
        self.Long_Length = np.int32((self.Xlim_end -  self.Xlim_start)/self.Grid_resol)
        self.Grid_Pos = np.zeros((2,self.Lat_Width,self.Long_Length))
        self.Grid_Pos[0,:,:] =  MGrid[0]
        self.Grid_Pos[1,:,:] =  MGrid[1]

    def convertRealXYToMapIdx(self, x, y):
        xIdx = (np.rint((x - self.mapXLim[0]) / self.unitGridSize)).astype(int)
        yIdx = (np.rint((y - self.mapYLim[0]) / self.unitGridSize)).astype(int)
        return xIdx, yIdx

    def UpdatePreviousInfo(self,Centre_in_robotF):
    	Occ_coord= np.where(self.Local_Map>0)
    	Vals = self.Local_Map[Occ_coord]
    	return Occ_coord, Vals
    
    def getMapPivotPoint(self):        
        Centre_in_robotF = np.where(self.dist_grid==np.min(self.dist_grid))
        return Centre_in_robotF
    
    def getScanMap(self,Meas_Z_t,Pose_X_t):
        (x,y,orientation) = (Pose_X_t[0] ,Pose_X_t[1] ,Pose_X_t[2])
        
        self.MapExpansionCheck(x,y)
        ScanMap = np.zeros((self.Lat_Width,self.Long_Length))
        
        dx = self.Grid_Pos.copy()
        dx[0, :, :] = np.float16(dx[0, :, :]) - x# A matrix of all the x coordinates of the cell
        dx[1, :, :] = np.float16(dx[1, :, :]) - y# A matrix of all the y coordinates of the cell
        theta_grid = np.array(np.rad2deg(np.arctan2(dx[1, :, :], dx[0, :, :])) - (orientation ))

        # Wrap to +pi / - pi
        theta_grid[theta_grid > 180] -= 360
        theta_grid[theta_grid < -180] += 360

        dist_grid = scipy.linalg.norm(dx, axis=0) # matrix of L2 distance to all cells from robot

        for i in Meas_Z_t.iterrows():
            r = i[1]['Range_XY_plane'] # range measured
            b = i[1]['Azimuth'] # bearing measured
            free_mask = (np.abs(theta_grid - b) <= self.FOV/2.0) & (dist_grid < (r - self.breath/2.0))
            occ_mask = (np.abs(theta_grid - b) <= self.FOV/2.0) & (np.abs(dist_grid - r) <= self.breath/2.0)

            # Adjust the cells appropriately
            ScanMap[occ_mask] += self.l_occ
            ScanMap[free_mask] += self.l_free 
            
        centre_pos = np.where(dist_grid==np.min(dist_grid))
        c_pos = (centre_pos[0].tolist(), centre_pos[1].tolist())
        
        #self.PlotMap(ScanMap, Pose_X_t,'LocalMap')
        return ScanMap,c_pos, dist_grid, theta_grid
        