import numpy as np
import os
from sklearn.metrics import mean_squared_error
from math import sqrt

def hx_x(satets):  # states=x,y,z,sigma,pho,beta
    '''
    :return: only measure x
    '''
    return np.array([satets[0]])

def lorenz63(states, dt):
    x, y, z = states
    dx = 10 * (y - x) * dt
    dy = (x * (28 - z) - y) * dt
    dz = (x * y - 8./3. * z) * dt
    return np.array([x + dx, y + dy, z + dz])

def load_value(path):
    values = np.load(path)
    #不返回初始时刻的观测
    return values[1:]

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# def Lorenz63(states, dt):
#     x, y, z, sigma, pho, beta = states
#     dx = sigma * (y - x) * dt
#     dy = (x * (pho - z) - y) * dt
#     dz = (x * y - beta * z) * dt
#     return np.array([x + dx, y + dy, z + dz, sigma, pho, beta])

def convert_w(W):
    for i in range(np.shape(W)[0]):
        keep = np.nonzero(W[i])
        W[i, keep] = 1 / len(keep[0])

    return W

def sqrt_mean_squared_error_network(true, pres):
    rmse_list = list()
    for i in range(np.shape(pres)[1]):
        rmse_list.append(sqrt(mean_squared_error(true, pres[:, i])))
    return rmse_list



def main():
    # W = np.load('../graph/W_er_0.2.npy')
    # convert = convert_w(W)
    # print(convert)
    true = np.array([1,2,3])
    pres = np.array([[1,2,3],[1,3,4]])
    print(mean_squared_error_network(true, pres))


if __name__ == '__main__':
    main()