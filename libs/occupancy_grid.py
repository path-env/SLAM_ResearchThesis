# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 09:14:05 2020

@author: Mangal
"""
import numpy as np
from numpy.core.arrayprint import _array_repr_implementation
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from scipy.ndimage.interpolation import map_coordinates

from libs.remove_ground_plane import RANSAC,Zbased
from utils.tools import Lidar_3D_Preprocessing, rotate
plt.ion()
class Map():
    def __init__(self, Poses=None ,MapMode = 2,sceneName = None):
        fig,self.ax = plt.subplots()
        self.breath = 1.2
        self.FOV = 1
        self.max_lidar_r = 30 # For carla
        self.l_occ = np.log(0.65/0.35)
        self.l_free = np.log(0.35/0.65)
        self.l_unknown = 0.5
        self.MapMode = MapMode
        self.Roffset = 5
        self.SceneName = sceneName
        self.Grid_resol = 1 # a x a m cell
        self.Angular_resol = np.rad2deg(np.arctan( self.Grid_resol/ self.max_lidar_r))
        self.Pose_t_1 = [0,0,0]
        self.MapIdx_G = np.array([[],[]])
        self.MapDim = np.int32((self.max_lidar_r+self.Roffset)/self.Grid_resol)
        if self.MapMode ==1: #Local Map
            self.Xlim_start = 0
            self.Xlim_end = (2*self.max_lidar_r + 91)
            self.Ylim_start = (0)
            self.Ylim_end = (2*self.max_lidar_r + 91)
            self.Lat_Width = np.int32((self.Ylim_end -  self.Ylim_start)/self.Grid_resol)
            self.Long_Length = np.int32((self.Xlim_end -  self.Xlim_start)/self.Grid_resol)
        else: #Global Map
            self.Xlim_start = -self.MapDim - (90/self.Grid_resol)
            self.Xlim_end = self.MapDim + (91/self.Grid_resol)
            self.Ylim_start = -self.MapDim - (90/self.Grid_resol)
            self.Ylim_end = self.MapDim + (91/self.Grid_resol)
            self.Lat_Width = np.int32((self.Ylim_end -  self.Ylim_start))
            self.Long_Length = np.int32((self.Xlim_end -  self.Xlim_start))
            
        self._createGrid()
        self.Local_Map = np.zeros(())
        cc = ( 0.5 * np.ones((self.Long_Length,self.Lat_Width)))
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
        InvModel = self.__inverse_sensor_model2(Pose_X_t, Meas_Z_t,False)
        #self.LO_t_i = np.log(np.divide(InvModel,np.subtract(1,InvModel)))  + self.LO_t_i  - self.LO_t
        self.LO_t_i = InvModel + self.LO_t_i  - self.LO_t
        #Plotting
        if PltEnable == True:
            self.PlotMap(self.LO_t_i, Pose_X_t, 'GlobalMap', self.Lat_Width, self.Long_Length)
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
        
        for row_ind in range(self.Xlim_start, self.Xlim_end):
            for col_ind in range(self.Ylim_start, self.Ylim_end):
                R2cell = np.hypot((row_ind - x),(col_ind - y))
                Azi2cell = np.degrees(np.arctan2((col_ind - y),(row_ind-x))) -orientation
                Almostthere = np.argmin(np.abs(np.subtract(Azi2cell,Azi_vec)))   
                Range_meas = Range_vec[Almostthere]
                Azi_meas = Azi_vec[Almostthere]
                if R2cell > min(self.max_lidar_r,(Range_meas +(self.breath/2))) or (abs(Azi2cell-Azi_meas) >self.FOV/2):
                    self.Local_Map[row_ind, col_ind] = self.l_unknown
                    continue
                if Range_meas < self.max_lidar_r and  (abs(R2cell - Range_meas)<(self.breath/2)):
                    self.Local_Map[row_ind, col_ind] =  self.l_occ
                    continue
                if R2cell < Range_meas:
                    self.Local_Map[row_ind, col_ind] = self.l_free
        #Plotting
        if PltEnable == True:
        #    self.PlotMap(self.Local_Map, Pose_X_t,'LocalMap', self.Lat_Width, self.Long_Length)
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
        MapLim = self.MapDim*2
        self.Local_Map = np.zeros((MapLim,MapLim))
        #self.Local_Map = np.zeros((self.Long_Length, self.Lat_Width))
        MGrid = np.meshgrid( np.arange(-self.MapDim, self.MapDim), 
                            np.arange(-self.MapDim, self.MapDim))
        Grid_Pos = np.zeros((2,MapLim,MapLim))
        Grid_Pos[0,:,:] =  MGrid[0]
        Grid_Pos[1,:,:] =  MGrid[1]
        dx = Grid_Pos.copy()
        dx[0, :, :] = np.float16(dx[0, :, :]) 
        dx[1, :, :] = np.float16(dx[1, :, :])
        theta_grid = np.rad2deg(np.arctan2(dx[1, :, :], dx[0, :, :])) #-orientation
        #Meas_Z_t =self._Lidar2MapFrame(Meas_Z_t, Pose_X_t)
        # Wrap to +pi / - pi
        theta_grid[theta_grid > 180] -= 360
        theta_grid[theta_grid < -180] += 360
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        self.dist_grid = sp.linalg.norm(dx, axis=0) # matrix of L2 distance to all cells from robot
        for i in Meas_Z_t.iterrows():
            rng = i[1]['Range_XY_plane'] # range measured
            azi = i[1]['Azimuth'] # bearing measured
            if rng > 45:
                free_mask = (np.abs(theta_grid - azi) <= self.FOV/2.0) & (self.dist_grid < (rng - self.breath/2.0))
                self.Local_Map[free_mask] += self.l_free
                continue
            
            free_mask = (np.abs(theta_grid - azi) <= self.FOV/2.0) & (self.dist_grid < (rng - self.breath/2.0))
            occ_mask = (np.abs(theta_grid - azi) <= self.FOV/2.0) & (np.abs(self.dist_grid - rng) <= self.breath/2.0)

            # Adjust the cells appropriately
            self.Local_Map[occ_mask] += self.l_occ
            self.Local_Map[free_mask] += self.l_free
        # self.Local_Map = sp.ndimage.rotate(self.Local_Map, orientation , reshape=False)
        # self.Local_Map = np.fliplr(self.Local_Map)
        # self.Local_Map = np.flipud(self.Local_Map)
        #Plotting
        if PltEnable == True:
           self.PlotMap(self.Local_Map, Pose_X_t,'LocalMap', self.Lat_Width, self.Long_Length)
        GlobalMap, self.MapIdx_G  =self.Lidar2MapFrame(self.Local_Map, Pose_X_t)
        self.Pose_t_1 = Pose_X_t
        return GlobalMap
    
    def Lidar2MapFrame(self, Local_Map , Pose_X_t):
        GlobalMap = np.zeros((self.Long_Length, self.Lat_Width))
        (y,x) = np.where(Local_Map > 0)
        Pos = np.array([Pose_X_t[1], Pose_X_t[0]]).reshape(2,1)
        # Pos = np.ceil(Pos).astype(np.int32)
        Meas = np.vstack((y,x))-self.MapDim
        # Offst = np.array([self.Long_Length, self.Lat_Width]).reshape(2,1)/2
        # Offst = Offst.astype(np.int32)
        MapIdx = rotate(-Pose_X_t[2]) @ Meas +Pos +self.MapDim
        # self.MapIdx_G = np.unique(np.concatenate((self.MapIdx_G,MapIdx-self.MapDim), axis=1),axis=1) # Grabs all theoccupied cells in the GlobalMap        
        MapIdx_G = np.unique(MapIdx,axis=1) # Stores the coords of only the last identified occupied 
        # MapIdx_G[1,:] = MapIdx_G[1,:] + 100
        MapIdx = np.ceil(MapIdx).astype(np.int32)
        GlobalMap[MapIdx[1], MapIdx[0]+100]= Local_Map[y,x]
        # self.PlotMap(GlobalMap,Pose_X_t,'Transformed to MAP Frame', self.Lat_Width, self.Long_Length)
        return GlobalMap, MapIdx_G

    def PlotMap(self,Map,Pose_X_t,title,lat_lim,long_lim):
        Veh = patches.Rectangle((Pose_X_t[0]+self.MapDim-5 , Pose_X_t[1]+100+self.MapDim-3.5),5,3.5, Pose_X_t[2],linewidth= 0.5, edgecolor='r')          
        probMap = np.exp(Map)/(1.+np.exp(Map))
        plt.title(f"{title} x:{np.round(Pose_X_t[0],5)} , y:{np.round(Pose_X_t[1],5)}, yaw:{np.round(Pose_X_t[2],5)}")
        plt.ylim(0,lat_lim)
        plt.xlim(0,long_lim)
        self.ax.add_patch(Veh)                
        plt.imshow(probMap.T, cmap='Greys')
        #plt.savefig('/Local/Local{title}.png')        
        plt.pause(0.001)
        #plt.matshow(probMap.T)
    
    def MapExpansionCheck(self, x, y):
        X_Lim = np.array([np.round(x-self.MapDim) , np.round(x+self.MapDim)])
        Y_Lim = np.array([np.round(y-self.MapDim) , np.round(y+self.MapDim)])
        quadrant,_,_ =  self.ExpansionDirection(X_Lim, Y_Lim)
        while (quadrant != -1):
            self.logger.info('Exnding the Map in quadrant=%d',quadrant)
            #self.expandOccupancyGrid(quadrant)
            quadrant, X_Lim, Y_Lim = self.ExpansionDirection(X_Lim, Y_Lim)
            if quadrant ==-1:
                self._createGrid()

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
            quadrant = -1
        return quadrant, X_Lim, Y_Lim
   
    def ExpandQuadrant(self, Expansionmode, X_Lim, Y_Lim):
        if Expansionmode ==1:
            pad_w = np.abs(X_Lim[0] - self.Xlim_start)
            pad_w = pad_w.astype(np.int32)
            self.LO_t_i = np.pad(self.LO_t_i, ((pad_w,0),(0,0)), 'constant', constant_values=(0,0))
            self.Xlim_start = np.int64(X_Lim[0])
        if Expansionmode ==2:
            pad_w = np.abs(X_Lim[1] - self.Xlim_end)
            pad_w = pad_w.astype(np.int32)
            self.LO_t_i = np.pad(self.LO_t_i, ((0,pad_w),(0,0)), 'constant', constant_values=(0,0))
            self.Xlim_end = np.int64(X_Lim[1])
        if Expansionmode ==3:
            pad_w = np.abs(Y_Lim[0] - self.Ylim_start)
            pad_w = pad_w.astype(np.int32)
            self.LO_t_i = np.pad(self.LO_t_i, ((0,0),(0,pad_w)), 'constant', constant_values=(0,0))
            self.Ylim_start = np.int64(Y_Lim[0])
        if Expansionmode ==4:
            pad_w = np.abs(Y_Lim[1] - self.Ylim_end)
            pad_w = pad_w.astype(np.int32)
            self.LO_t_i = np.pad(self.LO_t_i, ((0,0),(pad_w,0)), 'constant', constant_values=(0,0))
            self.Ylim_end = np.int64(Y_Lim[1])

    def _createGrid(self):
        MGrid = np.meshgrid( np.arange(self.Ylim_start,self.Ylim_end), 
                            np.arange(self.Xlim_start,self.Xlim_end))
        self.Lat_Width = np.int32((self.Ylim_end -  self.Ylim_start))
        self.Long_Length = np.int32((self.Xlim_end -  self.Xlim_start))
        self.Grid_Pos = np.zeros((2,self.Long_Length,self.Lat_Width))
        self.Grid_Pos[0,:,:] =  MGrid[0]
        self.Grid_Pos[1,:,:] =  MGrid[1]
        self.LO_t = np.zeros(( self.Long_Length,self.Lat_Width))

    def UpdatePreviousInfo(self,Centre_in_robotF):
    	Occ_coord= np.where(self.Local_Map>0)
    	Vals = self.Local_Map[Occ_coord]
    	return Occ_coord, Vals
    
    def getMapPivotPoint(self):        
        Centre_in_robotF = np.where(self.dist_grid==np.min(self.dist_grid))
        return Centre_in_robotF
    
    def getExtractMap(self,Pose_X_t):
        (y,x) = (self.MapDim, self.MapDim)
        Pos = np.array([Pose_X_t[1], Pose_X_t[0]]).reshape(2,1)
        Pos = np.ceil(Pos).astype(np.int32)
        centre_pos = self.getMapPivotPoint()
        Meas = np.vstack((centre_pos[1],centre_pos[0]))-self.MapDim
        MapIdx = rotate(-Pose_X_t[2]) @ Meas +Pos +self.MapDim
        MapIdx = np.ceil(MapIdx).astype(np.int32).flatten().tolist()
        x1, x2 = MapIdx[0]-self.MapDim+100 , MapIdx[0]+self.MapDim+100
        y1, y2 = MapIdx[1]-self.MapDim , MapIdx[1]+self.MapDim
        extract_map = self.LO_t_i[y1:y2, x1:x2]
        extract_map_enlarged = self._increaseResol(extract_map, Pose_X_t)
        extract_map_enlarged[extract_map_enlarged<0.8] = 0
        return extract_map, centre_pos 

    def getScanMap(self,Meas_Z_t,Pose_X_t):
        (x,y,orientation) = (Pose_X_t[0] ,Pose_X_t[1] ,Pose_X_t[2])
        #self.MapExpansionCheck(x,y)
        Mapsize = np.int32(self.MapDim)
        ScanMap = np.zeros((Mapsize*2,Mapsize*2))
        GG = np.zeros((2,Mapsize*2,Mapsize*2))
        MGrid = np.meshgrid( np.arange(-Mapsize,Mapsize), 
                    np.arange(-Mapsize,Mapsize))
        GG[0,:,:] =  MGrid[0]
        GG[1,:,:] =  MGrid[1]                    
        GG[0, :, :] = np.float16(GG[0, :, :]) 
        GG[1, :, :] = np.float16(GG[1, :, :]) 
        theta_grid = np.rad2deg(np.arctan2(GG[1, :, :], GG[0, :, :]))

        # Wrap to +pi / - pi
        theta_grid[theta_grid > 180] -= 360
        theta_grid[theta_grid < -180] += 360

        dist_grid = sp.linalg.norm(GG, axis=0)

        for i in Meas_Z_t.T:
            rng = np.hypot(i[0], i[1])
            azi =  np.rad2deg(np.arctan2(i[1] , i[0]))
            if rng > self.MapDim:
                free_mask = (np.abs(theta_grid - azi) <= self.FOV/2.0) & (dist_grid < (rng - self.breath/2.0))
                ScanMap[free_mask] += self.l_free
                continue
            free_mask = (np.abs(theta_grid - azi) <= self.FOV/2.0) & (dist_grid < (rng - self.breath/2.0))
            occ_mask = (np.abs(theta_grid - azi) <= self.FOV/2.0) & (np.abs(dist_grid - rng) <= self.breath/2.0)

            ScanMap[occ_mask] += self.l_occ
            ScanMap[free_mask] += self.l_free 
            
        centre_pos = np.where(dist_grid==np.min(dist_grid))
        c_pos = (centre_pos[0].tolist(), centre_pos[1].tolist())
        #ScanMap =self.Lidar2MapFrame(ScanMap, Pose_X_t)
        #self.PlotMap(ScanMap, Pose_X_t,'LocalMap', Mapsize*2, Mapsize*2)
        # ScanMap = self._increaseResol(ScanMap, Pose_X_t)
        return ScanMap,c_pos, dist_grid, theta_grid
        
    def getOccIndies_G(self):
        return self.MapIdx_G
    
    def _increaseResol(self,Map, Pose_X_t):
        resol = 0.01
        new_dims = []
        lat = np.int32(Map.shape[0]/resol)
        long = np.int32(Map.shape[1]/resol)
        Map[Map<0.8] = 0
        for original_length, new_length in zip(Map.shape, (lat,long)):
            new_dims.append(np.linspace(0, original_length-1, new_length))
        coords = np.meshgrid(*new_dims, indexing='ij')
        Map_enlarged = map_coordinates(Map, coords, order=0)
        Map_enlarged[Map_enlarged<0.8] = 0
        # self.PlotMap(Map_enlarged, Pose_X_t,'EnlargedMap',lat,long)
        return Map_enlarged