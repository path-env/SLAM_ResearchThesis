import numpy as np

from libs.occupancy_grid import Map
from libs.motion_models import CTRA_Motion_Model, VMM
#from libs.observation_models import Likelihood_Field_Observation_Model
from libs.scan_matching import ICP, RTCSM
class Particle:
    def __init__(self, Meas_X_t):
        #logger.info('Initializing PF values')
        self.id= 0
        #State Variables
        # One Particle initialized --With GPS data?
        self.x = Meas_X_t['x'] + np.random.randn()
        self.y = Meas_X_t['y']+ np.random.randn()
        self.yaw = Meas_X_t['yaw']+ np.random.randn()
        self.v = Meas_X_t['v']
        self.st = np.array([self.x,self.y,self.yaw,self.v]) # x,y,yaw,v  
        self.Est_X_t_1 = []
        #map
        self.m = [0]
        #weight
        self.w = 0.1
        #Gaus Approx
        self.mu = np.array([0.,0.,0.,0.])
        self.sigma = np.array([0.,0.,0.,0.])
        self.norm = 0.
        #trajectory
        self.x_traject = []
        self.y_traject = []
        #scan matcher
        self.SM = ICP()
        
    def motion_prediction(self, cmdIn, dt):
        self.Est_X_t = CTRA_Motion_Model(self.st, cmdIn, dt)
        self.Est_X_t_1 = self.st
        return self.Est_X_t
    
    def scan_match(self, Est_X_t, Meas_Z_t,Meas_Z_t_1):
        #Est_X_t = np.array([Est_X_t['x'],  Est_X_t['y'], Est_X_t['yaw'] ])
        GlobalTrans,RelativeTrans = self.SM.match(Meas_Z_t.to_numpy().T, 
                                        Meas_Z_t_1.to_numpy().T, 
                                        Est_X_t,
                                        self.st)
        return GlobalTrans, RelativeTrans