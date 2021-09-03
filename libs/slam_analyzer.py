import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
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
        self._init_plots()
    
    def set_groundtruth(self, GPS_Z_t, IMU_Z_t, Meas_X_t):
        self.True_x.append(GPS_Z_t['long'])
        self.True_y.append(GPS_Z_t['lat'])
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
        self.axs[0,0].plot(self.True_x,self.True_y,'g.', markersize=1)
        
        #prediction
        self.axs[0,1].plot(self.predict_x,self.predict_y,'b.' ,label='predicted', markersize=1)
        self.axs[0,1].plot(self.odom_x[1:], self.odom_y[1:], 'k*', label='Odom', markersize=1)
        #correction
        self.axs[0,1].plot(self.corrected_x, self.corrected_y, 'r+',label='corrected', markersize=1)

        # steer
        self.axs[1,0].plot(self.steer,'g.',label='steer', markersize=1)
        #Orientation
        self.axs[1,1].plot(self.True_yaw,'g.',label='GT', markersize=1)        
        self.axs[1,1].plot(self.predict_yaw,'b.' ,label='predicted', markersize=1)
        self.axs[1,1].plot(self.odom_yaw, 'k*', label='Odom', markersize=1)
        self.axs[1,1].plot(self.corrected_yaw,'r+',label='corrected', markersize=1)
        
        #Vel
        self.axs[0,2].plot(self.True_v,'g.',label='GT', markersize=1)  
        self.axs[0,2].plot(self.corrected_v,'r+',label='corrected', markersize=1)
        self.axs[0,2].plot(self.predict_v,'b.' ,label='predicted', markersize=1)
        
        #acc
        self.axs[1,2].plot(self.True_acc,'g.',label='GT', markersize=1)  
        plt.pause(0.0001)
        # plt.savefig('./results/GT_2D.png')
        plt.show()

    def _init_plots(self):
        self.axs[0,0].grid('on')
        self.axs[0,0].set_xlabel('Longitude')
        self.axs[0,0].set_ylabel('Latitude')
        self.axs[0,0].set_title("GPS_XY")
        self.axs[0,0].plot([],[],'g.', markersize=1)
        #self.axs[0,0].legend(loc='best')
        
        #prediction
        #self.axs[0,1].scatter(self.predict_x,self.predict_y,'r.' ,label='predicted', markersize=1)
        self.axs[0,1].plot([], [], 'k*', label='Odom', markersize=1)
        #correction
        self.axs[0,1].plot([], [], 'r+',label='corrected', markersize=1)
        self.axs[0,1].set_title("Odom-SM_XY")
        #self.axs[0,1].legend(loc='best')     
        self.axs[0,1].grid('on')

        # Orientation Comparison
        self.axs[1,0].set_title('GT_SteeringAngle')
        self.axs[1,0].grid('on')
        self.axs[1,0].set_ylabel('degrees')
        self.axs[1,0].plot([],[],'b.',label='predicted', markersize=1)
        #self.axs[1,0].legend(loc='best')
        
        self.axs[1,1].grid('on')
        self.axs[1,1].set_title('Odom-GPS-SM_Yaw(degreees)')
        #self.axs[1,1].legend(loc='best')
        self.axs[1,1].plot([],[],'g.',label='GT', markersize=1)        
        #self.axs[1,1].scatter(self.predict_yaw,'r.' ,label='predicted', markersize=1)
        self.axs[1,1].plot([],[], 'k*', label='Odom', markersize=1)
        self.axs[1,1].plot([],[],'r+',label='corrected', markersize=1)
        self.axs[1,1].plot([],[],'b.',label='predicted', markersize=1)
        
        #Vel
        self.axs[0,2].grid('on')
        self.axs[0,2].set_title("GT-SM_vel(m/s)")
        self.axs[0,2].plot([],[],'g.',label='GT', markersize=1)  
        self.axs[0,2].plot([],[],'r+',label='corrected', markersize=1)
        #acc
        self.axs[1,2].grid('on')
        self.axs[1,2].set_title("GT_acc(m/s^2)")
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

    def error_metrics(self):
        # corrected vs predicted
        # mean difference of location
        plt.figure()
        pos_diff_x = np.subtract(self.corrected_x, np.mean(self.predict_x))
        pos_diff_y = np.subtract(self.corrected_y, np.mean(self.predict_y))
        yaw_diff = np.subtract(self.corrected_yaw, np.mean(self.predict_yaw))
        plt.plot(pos_diff_x,legend = 'X_diff')
        plt.plot(pos_diff_y,legend = 'Y_diff')
        plt.plot(yaw_diff,legend = 'Yaw_diff')
        plt.legend(loc = 'best')
        print(f"The mean of diff b/t prediction and correction: X:{np.mean(pos_diff_x)}, Y:{np.mean(pos_diff_y)},Yaw:{np.mean(yaw_diff)}")

        # corrected vs odom
        plt.figure()
        pos_diff_x = np.subtract(self.corrected_x, self.odom_x)
        pos_diff_y = np.subtract(self.corrected_y, self.odom_y)
        yaw_diff = np.subtract(self.corrected_yaw, self.odom_yaw)
        plt.plot(pos_diff_x,legend = 'X_diff')
        plt.plot(pos_diff_y,legend = 'Y_diff')
        plt.plot(yaw_diff,legend = 'Yaw_diff')
        plt.legend(loc = 'best')
        print(f"The mean of diff b/t Odom and corrected: X:{np.mean(pos_diff_x)}, Y:{np.mean(pos_diff_y)},Yaw:{np.mean(yaw_diff)}")
       