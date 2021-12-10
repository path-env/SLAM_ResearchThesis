import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from numpy.core.defchararray import translate
from utils.tools import negativeposecomposition
#import matplotlib.animation as ani
#from matplotlib.animation import FuncAnimation
#from matplotlib import style
#style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 7})
plt.ion()

class Analyze:
    def __init__(self, title):
        self.fig, self.axs = plt.subplots(2,3, figsize=(15,10))
        self.fig.suptitle(title, fontsize=16)
        self.predict_x,self.predict_y,self.predict_yaw, self.predict_v = [], [], [], []
        self.corrected_x,self.corrected_y,self.corrected_yaw, self.corrected_v = [], [], [], []
        self.True_x,self.True_y, self.True_yaw, self.True_v, self.True_acc = [], [], [],[], []
        self.odom_x, self.odom_y, self.odom_yaw = [],[],[]
        self.steer = []
        self.time = []
        self.title = title
        self._init_plots()
    
    def set_groundtruth(self, GPS_Z_t, IMU_Z_t, Meas_X_t):
        self.True_x.append(GPS_Z_t[0])
        self.True_y.append(GPS_Z_t[1])
        self.True_yaw.append(IMU_Z_t['yaw'])
        self.True_v.append( Meas_X_t['v'])
        self.True_acc.append( Meas_X_t['acc'])
        self.odom_x.append( Meas_X_t['x'])
        self.odom_y.append( Meas_X_t['y'])
        self.odom_yaw.append( Meas_X_t['yaw'])
        self.steer.append(np.rad2deg(Meas_X_t['steer']))
        self.time.append(Meas_X_t['t'])
        
    def _set_trajectory(self,Est_X_t,st_prime):
        self.predict_x.append(st_prime[0])
        self.predict_y.append(st_prime[1])
        self.predict_yaw.append(st_prime[2])
        self.predict_v.append(st_prime[3])
        self.corrected_x.append(Est_X_t[0])
        self.corrected_y.append(Est_X_t[1])
        self.corrected_yaw.append(Est_X_t[2])
        self.corrected_v.append(Est_X_t[3])
        
    def plot_results(self):
        #axs[0,0].plot(self.corrected_x, self.corrected_y, 'r+',label='corrected', markersize=1)
        self.axs[0,1].plot(self.True_x,self.True_y,'g.', markersize=1)
        self.axs[0,0].plot(self.True_x,self.True_y,'g.', markersize=1)
        
        #prediction
        self.axs[0,1].plot(self.predict_x,self.predict_y,'b.' ,label='Predicted (MM)', markersize=1)
        self.axs[0,1].plot(self.odom_x, self.odom_y, 'k*', label='Odom', markersize=1)
        #correction
        self.axs[0,1].plot(self.corrected_x, self.corrected_y, 'r+',label='Corrected (SM)', markersize=1)
        # self.axs[0,1].axis('square')

        # steer
        self.axs[1,0].plot(self.steer,'g.',label='steer', markersize=1)
        #Orientation
        self.axs[1,1].plot(self.True_yaw,'g.',label='Ground Truth (GT)', markersize=1)        
        self.axs[1,1].plot(self.predict_yaw,'b.' ,label='Predicted (MM)', markersize=1)
        self.axs[1,1].plot(self.odom_yaw, 'k*', label='Odom', markersize=1)
        self.axs[1,1].plot(self.corrected_yaw,'r+',label='Corrected (SM)', markersize=1)
        
        #Vel
        self.axs[0,2].plot(self.True_v,'g.',label='Ground Truth (GT)', markersize=1)  
        self.axs[0,2].plot(self.corrected_v,'r+',label='Corrected (SM)', markersize=1)
        self.axs[0,2].plot(self.predict_v,'b.' ,label='Predicted (MM)', markersize=1)
        
        #acc
        self.axs[1,2].plot(self.True_acc,'g.',label='Ground Truth (GT)', markersize=1)  
        plt.pause(0.0001)
        # plt.savefig('./results/GT_2D.png')
        plt.show()

    def _init_plots(self):
        self.axs[0,0].grid('on')
        self.axs[0,0].set_xlabel('Longitude (m)')
        self.axs[0,0].set_ylabel('Latitude (m)')
        self.axs[0,0].set_title("GPS_XY")
        self.axs[0,0].plot([],[],'g.', markersize=1)
        #self.axs[0,0].legend(loc='best')
        
        #prediction
        #self.axs[0,1].scatter(self.predict_x,self.predict_y,'r.' ,label='predicted', markersize=1)
        self.axs[0,1].plot([], [], 'k*', label='Odom', markersize=1)
        self.axs[0,1].set_xlabel('X (m)')
        self.axs[0,1].set_ylabel('Y (m)')
        #correction
        self.axs[0,1].plot([], [], 'r+',label='Corrected (SM)', markersize=1)
        self.axs[0,1].set_title("Odom-SM_XY")
        #self.axs[0,1].legend(loc='best')     
        self.axs[0,1].grid('on')

        # Orientation Comparison
        self.axs[1,0].set_title('GT_SteeringAngle')
        self.axs[1,0].set_xlabel('Simulation Time (s)')
        self.axs[1,0].grid('on')
        self.axs[1,0].set_ylabel('Steering Angle (degrees)')
        self.axs[1,0].plot([],[],'b.',label='Predicted (MM)', markersize=1)
        #self.axs[1,0].legend(loc='best')
        
        self.axs[1,1].grid('on')
        self.axs[1,1].set_title('Odom-GPS-SM_Yaw')
        self.axs[1,1].set_xlabel('Simulation Time (s)')
        self.axs[1,1].set_ylabel('Yaw (degrees)')
        #self.axs[1,1].legend(loc='best')
        self.axs[1,1].plot([],[],'g.',label='Ground Truth (GT)', markersize=1)        
        #self.axs[1,1].scatter(self.predict_yaw,'r.' ,label='predicted', markersize=1)
        self.axs[1,1].plot([],[], 'k*', label='Odom', markersize=1)
        self.axs[1,1].plot([],[],'r+',label='Corrected (SM)', markersize=1)
        self.axs[1,1].plot([],[],'b.',label='Predicted (MM)', markersize=1)
        
        #Vel
        self.axs[0,2].grid('on')
        self.axs[0,2].set_title("GT-SM_vel")
        self.axs[0,2].set_xlabel('Simulation Time (s)')
        self.axs[0,2].set_ylabel('Velocity (m/s)')
        self.axs[0,2].plot([],[],'g.',label='Ground Truth (GT)', markersize=1)  
        self.axs[0,2].plot([],[],'r+',label='Corrected (SM)', markersize=1)
        #acc
        self.axs[1,2].grid('on')
        self.axs[1,2].set_title("GT_acc")
        self.axs[1,2].set_xlabel('Simulation Time (s)')
        self.axs[1,2].set_ylabel("Acceleration (m/s^2)")
        self.axs[1,2].plot([],[],'g.',label='GT', markersize=1)  
        
        h,l = self.axs[1,1].get_legend_handles_labels()
        self.fig.legend(h,l, loc='upper left')
        plt.tight_layout()
    
    def _particle_trajectory(self):
        plt.figure()
        for i in range(self.Particle_Cnt):
            plt.plot(self.Particle_DFrame.at[i,'traject_x'],self.Particle_DFrame.at[i,'traject_y'],label = i)
        plt.legend(loc = 'best')
        plt.show()

    def error_plot(self):
        # corrected vs True
        # mean difference of location
        plt.savefig(f'{self.title}_PositionPlot')
        plt.close()
        plt.savefig(f'{self.title}_Map')
        plt.close()
        plt.figure()
        pos_diff_x = np.subtract(self.corrected_x, self.True_x)
        pos_diff_y = np.subtract(self.corrected_y, self.True_y)
        yaw_diff = np.subtract(self.corrected_yaw, self.True_yaw)
        plt.xlabel('Simulation Time (s)')
        plt.plot(pos_diff_x, 'r+' ,label = 'X_diff (m)')
        plt.plot(pos_diff_y, 'g+',label = 'Y_diff (m)')
        plt.plot(yaw_diff,  'b+', label = 'Yaw_diff (degrees)')
        plt.legend(loc = 'best')
        plt.savefig(f'{self.title}_True_vs_Crct')
        print(f"The mean of diff b/t True and corrected: X:{np.mean(pos_diff_x)}, Y:{np.mean(pos_diff_y)},Yaw:{np.mean(yaw_diff)}")

        # corrected vs odom
        plt.figure()
        pos_diff_x = np.subtract(self.corrected_x, self.odom_x)
        pos_diff_y = np.subtract(self.corrected_y, self.odom_y)
        yaw_diff = np.subtract(self.corrected_yaw, self.odom_yaw)
        plt.xlabel('Simulation Time (s)')
        plt.plot(pos_diff_x, 'ro' ,label = 'X_diff (m)')
        plt.plot(pos_diff_y, 'go',label = 'Y_diff (m)')
        plt.plot(yaw_diff,  'bo', label = 'Yaw_diff (degrees)')
        plt.legend(loc = 'best')
        plt.savefig(f'{self.title}_Odom_vs_Crct')
        print(f"The mean of diff b/t Odom and corrected: X:{np.mean(pos_diff_x)}, Y:{np.mean(pos_diff_y)},Yaw:{np.mean(yaw_diff)}")
    
    def analysis_metrics(self):
        # Mean squared error of the results
        mean_diff_x = np.mean((np.array(self.corrected_x)- np.array(self.True_x))**2)
        mean_diff_y = np.mean((np.array(self.corrected_y)- np.array(self.True_y))**2)
        mean_diff_yaw =  np.mean((np.array(self.corrected_yaw)- np.array(self.True_yaw))**2)
        print(f"The MSE : X:{mean_diff_x}, Y:{mean_diff_y},Yaw:{mean_diff_yaw}")

    def benchmark_metric(self):
        trans_err, ori_err = [],[]
        trans_sqerr, ori_sqerr = [],[]
        for i in range(1,self.corrected_v.__len__()):
            X1 = [self.corrected_x[i-1], self.corrected_y[i-1], self.corrected_yaw[i-1]]
            X2 = [self.corrected_x[i], self.corrected_y[i], self.corrected_yaw[i]]
            Y1 = [self.True_x[i-1], self.True_y[i-1], self.True_yaw[i-1]]
            Y2 = [self.True_x[i], self.True_y[i], self.True_yaw[i]]
            X = negativeposecomposition(X1,X2)
            Y = negativeposecomposition(Y1,Y2)
            transf = negativeposecomposition(X,Y)
            transl, rot = np.hypot(transf[0], transf[1]), transf[2]
            trans_err.append(transl)
            ori_err.append(rot)
            trans_sqerr.append(transl**2)
            ori_sqerr.append(rot**2)
        trans_std = np.std(trans_err)
        ori_std = np.std(ori_err)
        trans_err = np.mean(trans_err)
        ori_err = np.mean(ori_err)

        trans_sqstd = np.std(trans_sqerr)
        ori_sqstd = np.std(ori_sqerr)
        trans_sqerr = np.mean(trans_sqerr)
        ori_sqerr = np.mean(ori_sqerr)

        Abs_Error = {'trans':[trans_err, trans_std], 'ori':[ori_err, ori_std]}
        Sq_Error = {'trans':[trans_sqerr, trans_sqstd], 'ori':[ori_sqerr, ori_sqstd]}
        print(f"Absolute Error= Trans:{Abs_Error['trans']}, Ori:{Abs_Error['ori']}")
        print(f"Square Error= Trans:{Sq_Error['trans']}, Ori:{Sq_Error['ori']}")
