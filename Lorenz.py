import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class Lorenz:
    """
    It's a differential equations:
        dx/dt=s(y-x)
        dy/dt=rx-y-xz
        dz/dt=xy-bz
    parameters:[s, r, b]
    Given xt-1, yt-1, zt-1 and then calculate the state of the next time step.
    """

    def __init__(self, params, initial_states, delta_t=0.01):
        self.delta_t = delta_t
        self.s, self.r, self.b = params
        self.x, self.y, self.z = initial_states

    def get_parameter(self, arr):
        self.s, self.r, self.b = arr[:, 0], arr[:, 1], arr[:, 2]

    def update(self, dx, dy, dz):
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.z + dz

    def run(self):
        dx = (self.s * (self.y - self.x)) * self.delta_t
        dy = (self.r * self.x - self.y - self.x * self.z) *self.delta_t
        dz = (self.x * self.y - self.b * self.z) * self.delta_t

        self.update(dx, dy, dz)

        # return np.stack([self.s, self.r, self.b, self.x, self.y, self.z])
        return [self.s, self.r, self.b, self.x, self.y, self.z]


def main():
    # experiment and plot
    params = [10., 28., 8./3.]
    initial_states = [10., 20., 30.]
    delta_t = 0.01
    start = 0
    end = 400
    epoch = int((end - start) / delta_t)
    lorenz = Lorenz(params, initial_states)
    states = list()
    for i in range(epoch):
        x = lorenz.run()
        states.append([x[-3], x[-2], x[-1]])
    states = np.array(states)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca(projection='3d')  # 获取当前子图，指定三维模式
    ax.plot(states[:, 0], states[:, 1], states[:, 2], lw=1.0, color='b')  # 画轨迹1
    plt.title('Lorenz system, s={},r={},b={},$x_0$={},$y_0$={},$z_0$={}'.format(
        params[0], params[1], params[2], initial_states[0], initial_states[1], initial_states[2]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()


    # produce data

    # lorenz = Lorenz(np.array([[10., 28., 3., 2., 1., 0.]]))
    # s = []
    # for i in range(10000):
    #     x = lorenz.run()
    #     if i % 50 == 0:
    #         print(i)
    #         s.append([x[-3], x[-2], x[-1]])
    # s = np.array(s)
    # np.save("state.npy", s)
