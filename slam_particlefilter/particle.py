import numpy as np

from libs.occupancy_grid import Map
from libs.motion_models import CTRA_Motion_Model
#from libs.observation_models import Likelihood_Field_Observation_Model

class Particle:
    def __init__(self, Meas_X_t, Meas_Z_t,GPS_Z_t, OG,SM, cnt):
        #logger.info('Initializing PF values')
        self.id= cnt
        #State Variables
        # One Particle initialized --With GPS data?
        self.x = GPS_Z_t[0]  + np.random.randn()*0.01 
        self.y = GPS_Z_t[1]  + np.random.randn()*0.01
        self.yaw = Meas_X_t[2]+ np.random.randn()*0.01
        self.v = Meas_X_t[3]
        self.st = np.array([self.x,self.y,self.yaw,self.v]) # x,y,yaw,v  
        self.Est_X_t_1 = []
        #map
        self.m = [0]
        #weight
        self.w = 0.1
        #Gaus Approx
        self.mu = np.array([0.,0.,0.,0.])
        self.sigma = np.array((0.1,0.1,0.1,0.1))
        self.norm = 0.
        #trajectory
        self.x_traject = []
        self.y_traject = []
        #scan matcher
        self.SM = SM
        self.err_thresh = 0
        self.iteration=0
        # self.OG = OG
        
    def motion_prediction(self, cmdIn, dt):
        self.Est_X_t = CTRA_Motion_Model(self.st, cmdIn, dt)
        self.Est_X_t_1 = self.st
        return self.Est_X_t
    
    def scan_match(self, Est_X_t, Est_X_t_1,Meas_Z_t,Meas_Z_t_1, method):
        self.iteration+=1
        if method == 0:
        #ICP SVD
            GT,self.GT_Lst = self.SM.match_SVD(Meas_Z_t_1.to_numpy().T, 
                                            Meas_Z_t.to_numpy().T, 
                                            Est_X_t,
                                            self.st,threshold = 0.000001, Iter =100)
            self.err_thresh = 0.001
        elif method == 1:
        #ICP LS
            GT,self.GT_Lst = self.SM.match_LS(Meas_Z_t_1.to_numpy().T, 
                                            Meas_Z_t.to_numpy().T, 
                                            Est_X_t,
                                            self.st,threshold = 0.000001, Iter =100)
            self.err_thresh = 0.0025
        #RTCSM
        elif method == 2:
            GT,self.GT_Lst = self.SM.match(Meas_Z_t.to_numpy().T, 
                                            Meas_Z_t_1.to_numpy().T, 
                                            Est_X_t,
                                            Est_X_t_1, self.iteration)
            self.err_thresh = 1/0.050
        return GT, self.GT_Lst