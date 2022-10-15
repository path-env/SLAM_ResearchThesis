# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:00:54 2021

@author: MangalDeep
"""
# To resolve VS code error
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
# np.seterr(all = "raise") 
import pandas as pd

from libs.remove_ground_plane import Zbased, RANSAC

# lambda Functions
rotate = lambda phi: np.array([[np.cos(np.deg2rad(phi)), -np.sin(np.deg2rad(phi))],
                                [np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))]]).reshape(2,2)

translate = lambda x,y : np.array([x , y]).reshape(2,1)

def Lidar_3D_Preprocessing(Meas_Z_t):
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
    Req_Indices = Zbased(Meas_Z_t,-2,5)
    Meas_Z_t = {'x':Meas_Z_t['x'][Req_Indices] ,'y':Meas_Z_t['y'][Req_Indices], 'z':Meas_Z_t['z'][Req_Indices]}
    Meas =  pd.DataFrame(Meas_Z_t)
    Meas = Meas.assign(Azimuth = np.rad2deg(np.arctan2(Meas['y'].values , Meas['x'].values)))
    Meas = Meas.assign(Range_XY_plane = np.hypot(Meas['x'].values,Meas['y'].values))
    Meas  = Meas.sort_values(by=['Azimuth','Range_XY_plane'],ascending=False)
    Meas = Meas.round({'Azimuth':1})
    Meas = Meas.drop_duplicates(['Azimuth'])
    Meas_Z_t = Meas.reset_index(drop=True)
    return Meas_Z_t

def normalize(arr):
    if (np.max(arr) - np.min(arr)) !=0:
        ans = [(x - np.min(arr))/ (np.max(arr) - np.min(arr)) for x in arr]
    else:
        ans = arr
    return ans

def softmax(arr):
    return np.exp(arr)/ np.sum(np.exp(arr))

def TransMatrix(T, orientation):
    H = np.diag([1.,1.,1.])
    H[:2,:2] = rotate(orientation)
    H[:2,2] = T
    return H

def negativeposecomposition(P1, P2):
    Pose1 = TransMatrix(P1[:2], P1[2])
    Pose2 = TransMatrix(P2[:2], P2[2])
    P = np.linalg.inv(Pose2) @ Pose1
    Transf_Lst = np.append(P[:2,2],np.rad2deg(np.arctan2(P[1,0],P[0,0])))
    return Transf_Lst

def poseComposition(P1, P2):
    Pose1 = TransMatrix(P1[:2], P1[2])
    Pose2 = TransMatrix(P2[:2], P2[2])
    P = Pose1 @ Pose2
    Transf_Lst = np.append(P[:2,2],np.rad2deg(np.arctan2(P[1,0],P[0,0])))
    return Transf_Lst

def latlonhtoxyzwgs84(lat,lon,h):
    #lat, lon in degrees
    a=6378137.0             #radius a of earth in meters cfr WGS84
    b=6356752.3             #radius b of earth in meters cfr WGS84
    e2=1-(b**2/a**2)
    latr=np.deg2rad(lat)      #latitude in radians
    lonr=np.deg2rad(lon)         #longituede in radians
    Nphi=a/np.sqrt(1-e2*np.sin(latr)**2)
    x=(Nphi+h)*np.cos(latr)*np.cos(lonr)
    y=(Nphi+h)*np.cos(latr)*np.sin(lonr)
    z=(b**2/a**2*Nphi+h)*np.sin(latr)
    return([x,y,z])