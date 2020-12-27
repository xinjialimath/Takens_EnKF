from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.common import Q_discrete_white_noise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

global measurements
measurements = np.load('measurements_1_x.npy')
takens_x = np.load('takens_1_x.npy')
enkf_x = np.load('enkf_1_x.npy')
true_x = np.load('true_x.npy')

#观测
def hx(x):
    return state[time]

global state
state = takens_x

#模型
def fx(x, dt):
    return state[time]

#观测噪声的标准差
std_noise = 1

#初值
x = np.array([state[0]])

#状态协方差矩阵初始化
P = np.eye(1)

#ensemble kalman filter
enkf = EnKF(x=x, P=P, dim_z=1, dt=0.1, N=5,
         hx=hx, fx=fx)

#观测噪声
enkf.R *= std_noise**2

#系统噪声
enkf.Q = np.eye(1)*1
filter_result=[]

#方便写fx
global time
time = 0

#运行
for i in range(0, state.shape[0]):
    m = measurements[i]
    enkf.predict()
    enkf.update(np.asarray([m]))
    filter_result.append(enkf.x)
    time = time + 1


filter_result = [x[0] for x in filter_result]
plt.figure()
plt.plot(np.arange(380, 383.5, 0.1), true_x, marker='s', label='True',color='r')
plt.plot(np.arange(380, 383.5, 0.1), filter_result, marker='s', label='EnKF+Takens',color='b')
plt.plot(np.arange(380, 383.5, 0.1), takens_x, marker='s', label='Takens',color='g')
plt.plot(np.arange(380, 383.5, 0.1), enkf_x, marker='s', label='EnKF',color='orange')
plt.scatter(np.arange(380, 383.5, 0.1), measurements, marker='o',label='measure',color='black')
plt.legend()
plt.show()

enkf_rmse = np.sqrt(np.mean((true_x - enkf_x) ** 2))
takens_rmse = np.sqrt(np.mean((true_x - takens_x) ** 2))
enkf_takens_rmse = np.sqrt(np.mean((true_x - filter_result) ** 2))
measurements_rmse = np.sqrt(np.mean((true_x - measurements) ** 2))
print('enkf_rmse: {:.9f}'.format(enkf_rmse))
print('takens_rmse: {:.9f}'.format(takens_rmse))
print('enkf_takens_rmse: {:.9f}'.format(enkf_takens_rmse))
print('measurements_rmse: {:.9f}'.format(measurements_rmse))

