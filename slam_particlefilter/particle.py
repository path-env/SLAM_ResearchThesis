import numpy as np

from libs.occupancy_grid import Map
from libs.motion_models import CTRA_Motion_Model
#from libs.observation_models import Likelihood_Field_Observation_Model

class Particle:
    def __init__(self, Meas_X_t, Meas_Z_t, OG,SM, cnt):
        #logger.info('Initializing PF values')
        self.id= cnt
        #State Variables
        # One Particle initialized --With GPS data?
        self.x = Meas_X_t[0]  #+ np.random.randn()
        self.y = Meas_X_t[1]  #+ np.random.randn()
        self.yaw = Meas_X_t[2]#+ np.random.randn()
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
        # self.OG = OG
        
    def motion_prediction(self, cmdIn, dt):
        self.Est_X_t = CTRA_Motion_Model(self.st, cmdIn, dt)
        self.Est_X_t_1 = self.st
        return self.Est_X_t
    
    def scan_match(self, Est_X_t, Est_X_t_1,Meas_Z_t,Meas_Z_t_1):
        #ICP or Scan2Map
        GT,self.GT_Lst = self.SM.match_LS(Meas_Z_t.to_numpy().T, 
                                         Meas_Z_t_1.to_numpy().T, 
                                         Est_X_t,
                                         Est_X_t_1)
        #RTCSM
        # GT,self.GT_Lst = self.SM.match(Meas_Z_t, 
        #                                  Meas_Z_t_1, 
        #                                  Est_X_t,
        #                                  Est_X_t_1)                                        
        return GT, self.GT_Lst