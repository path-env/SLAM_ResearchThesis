# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 09:14:05 2020

@author: Mangal
"""
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from RemoveGroundPlane import RANSAC,Zbased

class Map():
    def __init__(self, Poses=None ,MapMode = 1,sceneName = None):
        self.breath = 3
        self.FOV = 5
        self.max_r = 50 # For carla
        self.mapSize =  2*self.max_r + 91
        self.l_occ = np.log(0.65/0.35)
        self.l_free = np.log(0.35/0.65)
        self.l_unknown = 0.5
        self.MapMode = MapMode
        self.SceneName = sceneName
        self.Grid_resol = 1 # a x a m cell
        self.Angular_resol = np.rad2deg(np.arctan( self.Grid_resol/ self.max_r))
        
        if MapMode ==1: #Local Map
            self.Xlim_start = np.int32(0)
            self.Xlim_end = np.int32(2*self.max_r + 91)
            self.Ylim_start = np.int32(0);
            self.Ylim_end = np.int32(2*self.max_r + 91)
            self.Lat_Width =  self.mapSize #np.int32((self.Ylim_end -  self.Ylim_start))
            self.Long_Length =  self.mapSize #np.int32((self.Xlim_end -  self.Xlim_start))
        else: #Global Map
            self.Grid_resol = 1
            self.Xlim_start = np.int32(min(Poses['x']) - self.max_r - 10)
            self.Xlim_end = np.int32(max(Poses['x']) + self.max_r + 10)
            self.Ylim_start = np.int32(min(Poses['y']) - self.max_r - 10)
            self.Ylim_end = np.int32(max(Poses['y']) + self.max_r + 10)    
            self.Lat_Width = np.int32( (self.Ylim_end -  self.Ylim_start)/self.Grid_resol)
            self.Long_Length = np.int32((self.Xlim_end -  self.Xlim_start)/self.Grid_resol)
            
        self.Local_Map = np.zeros((self.Long_Length,self.Lat_Width))
        self.Grid_Pos = np.array([np.tile(np.arange(0, self.Long_Length, 1)[:,None], (1, self.Lat_Width)),np.tile(np.arange(0,self.Lat_Width, 1)[:,None].T, (self.Long_Length, 1))])
        c = ( 0.5 * np.ones((self.Long_Length,self.Lat_Width)))
        self.LO_t = np.log(np.divide(c,np.subtract(1,c)))
        self.LO_t_i = self.LO_t   
    
    def __Lidar_3D_Preprocessing(self,Meas_Z_t):
        '''
        Parameters
        ----------
        Meas_Z_t : Dict with keys X, Y, Z , Ring Indices
            Input in cartisean coordinate system.
        Returns
        -------
        Meas_Z_t : Pandas
            DESCRIPTION:
            Find Azimuth -180 to 180 degrees and round it to 0.1
            Find Range in XY_plane
            Sort it with Azimuth and Range_XY_plane (descending) and remove the duplicates. Reset the indices last
        '''
        #RANSAC
        #GroundPlaneRemovedIndices = RANSAC(Meas_Z_t,2)
        Req_Indices = Zbased(Meas_Z_t,-2.2,5)
        Meas_Z_t = {'x':Meas_Z_t['x'][Req_Indices] ,'y':Meas_Z_t['y'][Req_Indices], 'z':Meas_Z_t['z'][Req_Indices]}
        Meas =  pd.DataFrame(Meas_Z_t)
        Meas = Meas.assign(Azimuth = np.rad2deg(np.arctan2(Meas['y'].values , Meas['x'].values)))
        Meas = Meas.assign(Range_XY_plane = np.hypot(Meas['x'].values,Meas['y'].values))
        Meas  = Meas.sort_values(by=['Azimuth','Range_XY_plane'],ascending=False)
        Meas = Meas.round({'Azimuth':1})
        Meas = Meas.drop_duplicates(['Azimuth'])
        Meas_Z_t = Meas.reset_index(drop=True)
        return Meas_Z_t
    
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
        #self.Local_Map = 0*self.Local_Map
        if 'Range_XY_plane' not in Meas_Z_t.keys():
            Meas_Z_t = self.__Lidar_3D_Preprocessing(Meas_Z_t)
        InvModel = self.__inverse_sensor_model2(Pose_X_t, Meas_Z_t,PltEnable)
        #self.LO_t_i = np.log(np.divide(InvModel,np.subtract(1,InvModel))) # + self.LO_t_i  - self.LO_t
        return InvModel
     
    def __inverse_sensor_model(self,Pose_X_t, Meas_Z_t,PltEnable = False):
        if self.MapMode ==1:
            (x,y,orientation) = (0,0 ,Pose_X_t['yaw'])
        else:
            (x,y,orientation) = (Pose_X_t['x'] ,Pose_X_t['y'] ,Pose_X_t['yaw'])

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
                if R2cell > min(self.max_r,(Range_meas +(self.breath/2))) or (abs(Azi2cell-Azi_meas) >self.FOV/2):
                    self.Local_Map[row_ind- (-Limits), col_ind-(-Limits)] = self.l_unknown
                    continue
                if Range_meas < self.max_r and  (abs(R2cell - Range_meas)<(self.breath/2)):
                    self.Local_Map[row_ind- (-Limits), col_ind-(-Limits)] =  self.l_occ
                    continue
                if R2cell < Range_meas:
                    self.Local_Map[row_ind- (-Limits), col_ind-(-Limits)] = self.l_free
        #Plotting
        if PltEnable == True:
            probMap = np.exp(self.Local_Map)/(1.+np.exp(self.Local_Map))                    
            plt.imshow(probMap, cmap='Greys')
            plt.title(f"x:{np.round(Pose_X_t['x'],5)} , y:{np.round(Pose_X_t['y'],5)}, yaw:{np.round(Pose_X_t['yaw'],5)}")
            plt.draw()
            plt.show() 
        self.Local_Map[Limits, Limits] = 2   #Location of the Lidar
        return self.Local_Map
        
    def __inverse_sensor_model2(self,Pose_X_t, Meas_Z_t,PltEnable = False):
        if self.MapMode ==1:
            (x,y,orientation) = (Pose_X_t['x'] ,Pose_X_t['y']  ,Pose_X_t['yaw'])
        else:
            (x,y,orientation) = (Pose_X_t['x'] ,Pose_X_t['y'] ,Pose_X_t['yaw'])

        dx = self.Grid_Pos.copy()
        dx[0, :, :] = np.float16(dx[0, :, :]) - x - np.int32(self.Long_Length/2)# A matrix of all the x coordinates of the cell
        dx[1, :, :] = np.float16(dx[1, :, :]) - y - np.int32(self.Lat_Width/2) # A matrix of all the y coordinates of the cell
        theta_to_grid = np.rad2deg(np.arctan2(dx[1, :, :], dx[0, :, :])) - (orientation )

        # Wrap to +pi / - pi
        theta_to_grid[theta_to_grid > 180] -= 360
        theta_to_grid[theta_to_grid < -180] += 360

        dist_to_grid = scipy.linalg.norm(dx, axis=0) # matrix of L2 distance to all cells from robot

        for i in Meas_Z_t.iterrows():
            r = i[1]['Range_XY_plane'] # range measured
            b = i[1]['Azimuth'] # bearing measured
            free_mask = (np.abs(theta_to_grid - b) <= self.FOV/2.0) & (dist_to_grid < (r - self.breath/2.0))
            occ_mask = (np.abs(theta_to_grid - b) <= self.FOV/2.0) & (np.abs(dist_to_grid - r) <= self.breath/2.0)

            # Adjust the cells appropriately
            self.Local_Map[occ_mask] += self.l_occ
            self.Local_Map[free_mask] += self.l_free
            
        #Plotting
        if PltEnable == True:
            probMap = np.exp(self.Local_Map)/(1.+np.exp(self.Local_Map)) 
            plt.title(f"x:{np.round(Pose_X_t['x'],5)} , y:{np.round(Pose_X_t['y'],5)}, yaw:{np.round(Pose_X_t['yaw'],5)}")                
            plt.imshow(probMap, cmap='Greys')
            plt.draw()
            plt.show() 
        return self.Local_Map
                   