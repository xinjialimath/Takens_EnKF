import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def lorenz(xyz, t, sets):
    sigma, rho, beta = sets
    x, y, z = xyz
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

def takens(x, dataset, K, dt=0.01):
    temp = []
    index = []
    n = dataset.shape[0]
    for i in range(n-1):
        dist = np.sum((dataset[i, :] - x) ** 2)
        temp.append([dist, i])
    temp.sort()
    for i in range(K):
        index.append(temp[i][1])
    index = [value + 1 for value in index]
    pred_x = np.mean(dataset[index, -1])
    pred_y = np.append(x[1:3], pred_x)
    return pred_y

#真解
t = np.arange(0, 40, 0.01)
data = odeint(lorenz, (10., 1., 0.), t, args=([10., 28., 8/3],))
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(x, y, z)
plt.title('Lorenz-63')
plt.figure()
plt.plot(t,x)
plt.title('X')
#重构真解
tau = 10
m = 3
N = len(x)
x1 = x[0 : N - (m - 1) * tau]
x2 = x[0 + tau : N - (m - 2) * tau]
x3 = x[0 + 2 * tau : N]
xxx = np.stack([x1, x2, x3], 1)
xxx_train = xxx[:int(xxx.shape[0]*0.95),:]
xxx_test = xxx[int(xxx.shape[0]*0.95):,:]
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(x1, x2, x3)
plt.title('Reconstruction')
plt.show()

# #观测
# std_noise = 1
# measurements = [value + np.random.normal(0,std_noise) for value in x]

# #重构观测
# measurements_x1 = measurements[0 : N - (m - 1) * tau]
# measurements_x2 = measurements[0 + tau : N - (m - 2) * tau]
# measurements_x3 = measurements[0 + 2 * tau : N]
# measurements_xxx = np.stack([measurements_x1, measurements_x2, measurements_x3], 1)
# measurements_xxx_train = measurements_xxx[:int(measurements_xxx.shape[0]*0.95),:]

# #用真解预测
# pred=[]
# pred_x=xxx_train[-1, :]
# for i in range(xxx_test.shape[0]):
#     pred_x = takens(pred_x, xxx_train, 5)
#     pred.append(pred_x)

# #用观测值预测
# measurements_pred=[]
# measurements_pred_x=measurements_xxx_train[-1, :]
# for i in range(xxx_test.shape[0]):
#     measurements_pred_x = takens(measurements_pred_x, measurements_xxx_train, 5)
#     measurements_pred.append(measurements_pred_x)


# #画图
# index = -xxx_test.shape[0]
# x_pred = [value[-1] for value in pred]
# measurements_x_pred = [value[-1] for value in measurements_pred]
# plt.figure()
# plt.plot(t[index:index+35], x[index:index+35], marker='.', label = 'true')
# plt.plot(t[index:index+35], x_pred[:35], marker='.', label = 'pred')
# plt.plot(t[index:index+35], measurements_x_pred[:35], marker='.', label = 'measurements_pred')
# plt.legend()
# plt.show()

# # np.save('true_x.npy', x[index:index+35])
# # np.save('measurements_1_x.npy', measurements[index:index+35])
# # np.save('takens_1_x.npy', measurements_x_pred[:35])
