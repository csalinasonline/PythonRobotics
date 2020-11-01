"""

Unscented kalman filter (UKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)
modified: Conrad Salinas

motion_model (state matrix) [4x1]:              2D x-y position, yaw, velocity 
                                                and yaw rate
                                            
observation_model (measurement matrix) [2x1]:   2D x-y position, 
                                                velocity and yaw rate


# N:                                  Number of time steps
# show_final:                         Flag for showing final result
show_animation:                     Flag for showing each animation frame
# show_ellipse:                       Flag for showing covariance ellipse

DT [1x1]:                          Duration of time step
nx [1x1]:                          Dimensions
zPred [2x1]:
xEst [4x1]:                        State estimate
PEst [4x4]:                        State estimate covariance
PPred [4x4]:
xPred [4x1]:
xTrue [4x1]:                       Ground truth value of state
xDR [4x1]:                         Dead Reckoning
y [2x1]:
yaw [1x1]:
v [1x1]:
u [2x1]: 
ud [2x1]:                           
F [4x4]:
B [4x2]:
H [2x4]:
K [4x2]:
x [4x1]:
xd [4x1]:    
z [2x1]:
Q [4x4]:                           Process noise covariance
R [2x2]:                           Sensor noise covariance
Pxy [2x2]:
Psqrt [4x4]:
P [4x4]:
Pi [4x4]:
eigval [2x1]:
eigvec [2x2]:
sigma [4x9]:
gamma [1x1]:
wc [1x9]:
wm [1x9]:
dx [4x9]:
dy [2x9]:
zb [4x1]:
z_sigma [2x9]:
st [2x2]:
Pxz [4x2]:
INPUT_NOISE [2x2]:
GPS_NOISE [2x2]: 
SIM_TIME [1X1]:

"""

#===== Imports


import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import scipy.linalg


#===== Constants


# Covariance for UKF simulation
# [4x4]
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
# [2x2]
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
# [2x2]
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
# [2x2]
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

#  UKF Parameter
ALPHA = 0.001
BETA = 2
KAPPA = 0

show_animation = True


#===== Common Implementation Methods


# out: [2x1]
def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    #[2x1]
    u = np.array([[v], [yawrate]])
    return u


# out: [4x1], out: [2x1], out: [4x1], out: [2x1]
# in:  [4x1], in: [4x1], in: [2x1]
def observation(xTrue, xd, u):
    # [4x1]
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    # [2x1] = [2x1] + [2x2] @ [2x1]
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    # [2x1] = [2x1] + [2x2] @ [2x1]
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    # [4x1]
    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


# out: [4x1]
# in:  [4x1], in: [2x1]
def motion_model(x, u):
    # [4x4]
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    # [4x2]
    B = np.array([[DT * math.cos(x[2]), 0],
                  [DT * math.sin(x[2]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    # [4x1] = [4x4] @ [4x1] + [4x2] @ [2x1]
    x = F @ x + B @ u

    return x


# out: [2x1]
# in:  [4x1]
def observation_model(x):
    # [2x4]
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    # [2x1] = [2x4] @ [4x1]
    z = H @ x

    return z


# in: [4x1], in: [4x4]
def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    # [2x2]
    Pxy = PEst[0:2, 0:2]
    # [2x1], [2x2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    # [64x1]
    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    # [1x1]
    a = math.sqrt(eigval[bigind])
    # [1x1]
    b = math.sqrt(eigval[smallind])
    # [64x1]
    x = [a * math.cos(it) for it in t]
    # [64x1]
    y = [b * math.sin(it) for it in t]
    # [1x1]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    # [2x2]
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    # [2x64] = [2x2] @ [2x64]
    fx = rot @ np.array([x, y])
    # [64x1] = ([1x64] + [1x1])
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    # [64x1] = ([1x64] + [1x1])    
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")

#===== Specific Algo Implementation Methods


# out: [4x9]
# in: [4x1], in: [4x4], in: [1x1]
def generate_sigma_points(xEst, PEst, gamma):
    # [4x1]
    sigma = xEst
    
    # [4x4]
    Psqrt = scipy.linalg.sqrtm(PEst)
    # n = 4
    n = len(xEst[:, 0])
    # Positive direction
    # [4x5]
    for i in range(n):
        # [4x(i+1)] = ([4xi], [4x1])
        sigma = np.hstack((sigma, xEst + gamma * Psqrt[:, i:i + 1]))

    # Negative direction
    # [4x9]
    for i in range(n):
        # [4x(i+1)] = ([4xi], [4x1])
        sigma = np.hstack((sigma, xEst - gamma * Psqrt[:, i:i + 1]))

    return sigma


# out: [4x9]
# in: [4x9], in: [2x1]
def predict_sigma_motion(sigma, u):
    """
        Sigma Points prediction with motion model
    """
    # [4x9]
    for i in range(sigma.shape[1]):
        sigma[:, i:i + 1] = motion_model(sigma[:, i:i + 1], u)
        
    return sigma


# out: [4x9]
# in: [4x9]

    return sigma


# out: [2x9]
# in: [4x9]
def predict_sigma_observation(sigma):
    """
        Sigma Points prediction with observation model
    """
    # [4x9]
    for i in range(sigma.shape[1]):
        # [2xi] = ([4xi])
        sigma[0:2, i] = observation_model(sigma[:, i])

    # [2x9]
    sigma = sigma[0:2, :]

    return sigma


# out: [4x4]
# in: [4x1], in: [4x9], in: [1x9], in: [4x4]
def calc_sigma_covariance(x, sigma, wc, Pi):
    # nSigma = 9
    nSigma = sigma.shape[1]
    # [4x9] = [4x9] - [4x1]
    d = sigma - x[0:sigma.shape[0]]
    # [4x4]
    P = Pi
    # [4x4]
    for i in range(nSigma):
        # [4x2] = [4x2] + [1x1] * [4xi] @ [ix2]
        P = P + wc[0, i] * d[:, i:i + 1] @ d[:, i:i + 1].T
    return P


# out: [4x2]
# in: [4x9], in: [4x1], in: [2x9], in: [4x1], in: [1x9]
def calc_pxz(sigma, x, z_sigma, zb, wc):
    # nSigma = 9
    nSigma = sigma.shape[1]
    # [4x9] = [4x9] - [4x1]
    dx = sigma - x
    # [2x9] = [2x9] - [2x1]
    dz = z_sigma - zb[0:2]
    # [4x2]
    P = np.zeros((dx.shape[0], dz.shape[0]))
    # [4x2]
    for i in range(nSigma):
        # [4x2] = [4x2] + [1x1] * [4xi] @ [ix2]
        P = P + wc[0, i] * dx[:, i:i + 1] @ dz[:, i:i + 1].T

    return P


# out: [4x1], out: [4x4]
# in: [4x1], in: [4x4], in: [2x1], in: [2x1], in: [1x9], in: [1x9], in: [1x1]
def ukf_estimation(xEst, PEst, z, u, wm, wc, gamma):
    #  Predict
    # [4x9] = ([4x1], [4x4], [1x1])
    sigma = generate_sigma_points(xEst, PEst, gamma)
    # [4x9]
    sigma = predict_sigma_motion(sigma, u)
    # [4x1] = ([1x9] @ [9x4]).T
    xPred = (wm @ sigma.T).T
    # [4x4] = ([4x1], [4x9], [1x9], [2x2])
    PPred = calc_sigma_covariance(xPred, sigma, wc, Q)

    #  Update
    # [4x1] = ([4x1])
    zPred = observation_model(xPred)
    # [2x1] = [2x1] - [2x1]
    y = z - zPred
    # [4x9] = ([4x1], [4x4], [1x1])
    sigma = generate_sigma_points(xPred, PPred, gamma)
    # [4x1] = ([1x9] @ [9x4]).T
    zb = (wm @ sigma.T).T
    # [2x9] = ([4x9])
    z_sigma = predict_sigma_observation(sigma)
    # [2x2] = ([4x1], [2x9], [1x9], [2x2])
    st = calc_sigma_covariance(zb, z_sigma, wc, R)
    # [4x2] = ([4x9], [4x1], [2x9], [4x1], [1x9])
    Pxz = calc_pxz(sigma, xPred, z_sigma, zb, wc)
    # [4x2] = [4x2] @ [2x2]
    K = Pxz @ np.linalg.inv(st)
    # [4x1] = [4x1] + [4x2] @ [2x1]
    xEst = xPred + K @ y
    # [4x4] = [4x4] - [4x2] @ [2x2] @ [2x4]
    PEst = PPred - K @ st @ K.T

    return xEst, PEst


# out: [1x9], out: [1x9], out: [1x1]
# in: [1x1]
def setup_ukf(nx):
    lamb = ALPHA ** 2 * (nx + KAPPA) - nx
    # calculate weights
    wm = [lamb / (lamb + nx)]
    wc = [(lamb / (lamb + nx)) + (1 - ALPHA ** 2 + BETA)]
    for i in range(2 * nx):
        wm.append(1.0 / (2 * (nx + lamb)))
        wc.append(1.0 / (2 * (nx + lamb)))
    gamma = math.sqrt(nx + lamb)

    # [1x9]
    wm = np.array([wm])
    # [1x9]
    wc = np.array([wc])

    return wm, wc, gamma


#===== Main Method


def main():
    print(__file__ + " start!!")

    nx = 4  # State Vector [x y yaw v]'
    # [4x1]
    xEst = np.zeros((nx, 1)) 
    # [4x1]
    xTrue = np.zeros((nx, 1))
    # [4x4]
    PEst = np.eye(nx)
    # [4x1]
    xDR = np.zeros((nx, 1))  # Dead reckoning

    # [1x9], [1x9], [1x1] = ([1x1])
    wm, wc, gamma = setup_ukf(nx)

    # history
    # [4x1]
    hxEst = xEst
    # [4x1]
    hxTrue = xTrue
    # [4x1]
    hxDR = xTrue
    # [2x1]
    hz = np.zeros((2, 1))

    time = 0.0

    while SIM_TIME >= time:
        time += DT
        # [2x1] = ()
        u = calc_input()

        # [4x1], [2x1], [4x1], [2x1] = ([4x1], [4x1], [2x1])
        xTrue, z, xDR, ud = observation(xTrue, xDR, u)
        
        # [4x1], [4x4] = ([4x1], [4x4], [2x1], [2x1], [1x9], [1x9], [1x1])
        xEst, PEst = ukf_estimation(xEst, PEst, z, ud, wm, wc, gamma)

        # store data history
        # [4x(n+1)] = ([4xn], [4x1])
        hxEst = np.hstack((hxEst, xEst))
        # [4x(n+1)] = ([4xn], [4x1])
        hxDR = np.hstack((hxDR, xDR))
        # [4x(n+1)] = ([4xn], [4x1])
        hxTrue = np.hstack((hxTrue, xTrue))
        # [2x(n+1)] = ([2xn], [2x1])
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


#===== Script Start


if __name__ == '__main__':
    main()
