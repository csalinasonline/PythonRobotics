"""

linear kalman filter (EKF) simple battery measurement example

author: Conrad Salinas

port: https://github.com/philbooks/Kalman-Filter-for-Beginners

===

show_animation:                    Flag for showing each animation frame

DT [1x1]:                          Duration of time step
SIM_TIME [1x1]:
BATT_VOLT [1x1]:
MEAN_ERR [1x1]:
STD_SER [1x1]:

x_init [1x1]:
P_init [1x1]:
Q [1x1]:                           Process noise covariance
R [1x1]:                           Sensor noise covariance
A [1x1]:
H [1x1]:
Pp [1x1]:                          Prediction of the error covariance
Pe [1x1]:                          Computation of the error covariance
K [1x1]:                           Kalman gain
xp [1x1]:
x_est [1x1]:


"""


#===== Imports


import math

import matplotlib.pyplot as plt
import numpy as np


#===== Constants


# [1x1]
Q = np.array([0]) # Process noise covariance
# [1x1]
R = np.array([4])  # Sensor noise covariance
# [1x1]
A = np.array([1]) 
# [1x1]
H = np.array([1]) 
# [1x1]
Pp = np.array([0]) # Prediction of the error covariance
# [1x1]
Pe = np.array([0]) # Computation of the error covariance
# [1x1]
K = np.array([0]) # Kalman gain
# [1x1]
x_init = np.array([14]) # inital measurement
# [1x1]
P_init = np.array([6]) # inital variance
# [1x1]
xp = np.array([0]) 
# [1x1]
x_est = np.array([0]) 

DT = 0.1  # time tick [s]
SIM_TIME = 3  # simulation time [s]
BATT_VOLT = 14.4
MEAN_ERR = 0
STD_ERR = 4

show_animation = True


#===== Common Implementation Methods


# out: [1x1]
def calc_input():
    # [1x1] = [1x1] + [1x1] + [1x1]
    w = MEAN_ERR + STD_ERR * np.random.randn(1, 1)
    # [1x1] = [1x1] + [1x1]
    z = BATT_VOLT + w

    return z

# out: [1x1], out: [1x1], out: [1x1]
# in: [1x1], in: [1x1], in: [1x1], in: [1x1]
def linear_kalman(z, x1, P1, first_run):
    if first_run:
        # 1. Prediction of estimate
        # [1x1] = [1x1]
        xp = x_init             
        # [1x1] = [1x1]
        # Prediction of the error covariance
        Pp = P_init    
    else: 
        # [1x1] = [1x1]
        xp = x1 
        # [1x1] = [1x1]
        Pp = P1
    
    # 2. Compution of Kalman gain
    # [1x1] = [1x1] * (1 / ([[1x1] + [1x1]))                            
    K = Pp * (1 / (Pp + R)) 
   
    # 3. Computaion of the estimate
    # [1x1] = [1x1] + [1x1] * ([1x1] - [1x1])
    x_est = xp + K * (z - xp)   
    
    # 4. Computaion of the error covariance
    # [1x1] = [1x1] - [1x1] * [1x1]
    Pe = Pp - K * Pp            
    
    return x_est, K, Pe


#===== Main Method


def main():
    print(__file__ + " start!!")

    nx = 1  # State Vector [1]
    # [1x1]
    xEst = np.zeros((nx, 1)) 
    # [1x1]
    xTrue = np.zeros((nx, 1))
    # [1x1]
    PEst = np.eye(nx)
    # [1x1]
    xDR = np.zeros((nx, 1))  

    # history
    # [1x1]
    hxTime = 0
    # [1x1]
    hxK = 0
    # [1x1]
    hxPEst = 0
    # [1x1]
    hxIdx = 0
    # [1x1]
    hxEst = xEst
    # [1x1]
    hxTrue = BATT_VOLT
    # [1x1]
    hxDR = xTrue
    
    time = 0.0
    first_run = True
    idx = 0

    while SIM_TIME >= time:
        time += DT
        # [2x1] = ()
        u = calc_input()

        if first_run:
             # [1x1], [1x1], [1x1] = ([1x1], [1x1], [1x1], [1x1])
             xEst, K, PEst = linear_kalman(u, xEst, PEst, first_run)
             first_run = False
        else:
             # [1x1], [1x1],[1x1] = ([1x1], [1x1], [1x1], [1x1])
             xEst, K, PEst = linear_kalman(u, xEst, PEst, first_run)
            
        #print(u, xEst, K, PEst)
        
        # store data history
        # [1x(n+1)] = ([1xn], [1x1])
        hxTrue = np.hstack((hxTrue, BATT_VOLT))
        # [1x(n+1)] = ([1xn], [1x1])
        hxTime = np.hstack((hxTime, time))
        # [1x(n+1)] = ([1xn], [1x1])
        hxK = np.hstack((hxK, K))
        # [1x(n+1)] = ([1xn], [1x1])
        hxPEst = np.hstack((hxPEst, PEst))
        # [1x(n+1)] = ([1xn], [1x1])
        hxIdx = np.hstack((hxIdx, idx))
        # [1x(n+1)] = ([1xn], [1x1])
        hxEst = np.hstack((hxEst, xEst))
        # [1x(n+1)] = ([1xn], [1x1])
        hxDR = np.hstack((hxDR, u))
        
        # 
        idx = idx + 1
        
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.subplot(211)
            plt.title('Measurements and Filter')
            plt.plot(hxTime, hxDR[:].flatten(), ".g")
            plt.plot(hxTime, hxEst[:].flatten(), "-r")
            plt.plot(hxTime, hxTrue, "--k")
            plt.legend(['Measurements','LKF','True'], 
                       bbox_to_anchor=(1.05, 1), 
                       loc='upper left', 
                       borderaxespad=0)
            plt.grid(True)
            plt.subplot(223)
            plt.title('Variance')
            plt.plot(hxTime, hxPEst, "-k")
            plt.grid(True)
            plt.subplot(224)
            plt.title('Kalman Gain')
            plt.plot(hxTime, hxK, "-b")
            plt.grid(True)
            plt.tight_layout()
            plt.pause(0.001)


#===== Script Start


if __name__ == '__main__':
    main()
