import numpy as np
import os
import time,datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from Lorenz import  Lorenz
from utils import lorenz63, hx_x, mkdir
from config import Config
from tqdm import tqdm

class Generate_Lorenz63_Data():
    def __init__(self, config):
        self.params = config.params #参数
        self.z_dim = config.z_dim # dimension of measurement
        self.x_dim = len(config.initial_states)
        self.initial_states = config.initial_states
        self.h = config.h
        self.dt = config.dt
        self.end = config.end
        self.R = config.R
        self.Q = config.Q


        self.t = np.arange(0, config.end, config.dt)
        self.epoch = int(config.end / config.dt) + 1

        self.states_clean_list = list()
        self.states_list = list()
        self.measures_list = list()
        self.states_params_list = list() #all_states = states (x,y,z) and params
        self.states_params_clean_list = list()

        self.gt_ode = None
        self.gt_Euler = None
        self.lorenz_dir = None
        self.figure_dir = None
        self.data_dir = None


    def initial_config_and_dirs(self, index):
        # dateArray = datetime.datetime.fromtimestamp(time.time())
        # self.begin_time = dateArray.strftime("%Y_%m_%d_%H_%M_%S")
        # self.lorenz_dir = '../Lorenz_data/' + self.begin_time
        self.lorenz_dir = '../Lorenz_data/' + str(index)
        self.figure_dir = os.path.join(self.lorenz_dir, 'figures')
        self.data_dir = os.path.join(self.lorenz_dir, 'data')
        mkdir(self.lorenz_dir)
        mkdir(self.figure_dir)
        mkdir(self.data_dir)
        #save config
        f = open(os.path.join(self.lorenz_dir, 'config.txt'), mode='a+')
        # config_dict = config.__dict__
        # for key,value in config_dict:
        #     if key in ['params', 'initial_states', 'dt', 'end', 'h', 'epoch']:
        #         f.writelines('{}: {}\r'.format(key,value))
        f.writelines('{}: {}\r'.format('params',self.params))
        f.writelines('{}: {}\r'.format('initial_states',self.initial_states))
        f.writelines('{}: {}\r'.format('dt',self.dt))
        f.writelines('{}: {}\r'.format('end',self.end))
        f.writelines('{}: {}\r'.format('h',self.h))
        f.writelines('{}: {}\r'.format('epoch',self.epoch))
        f.writelines('{}: {}\r'.format('R',self.R))
        f.writelines('{}: {}\r'.format('Q',self.Q))
        f.writelines('{}: {}\r'.format('z_dim',self.z_dim))
        f.close()

    def GT_from_odeint(self,save_name='lorenz63_gt_ode.jpg'):
        #Ground truth by odeint
        self.gt_ode = odeint(lorenz63, tuple(self.initial_states), self.t)
        x_ode = self.gt_ode[:, 0]
        y_ode = self.gt_ode[:, 1]
        z_ode = self.gt_ode[:, 2]

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(x_ode, y_ode, z_ode, 'b')
        plt.title('Lorenz63 system, ground truth from odeint, $x_0$={},$y_0$={},$z_0$={}'.format(
            self.initial_states[0], self.initial_states[1], self.initial_states[2]))
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.show()
        plt.savefig(os.path.join(self.figure_dir, save_name))
        plt.close()

    def GT_from_Euler(self,save_name='lorenz63_gt_Euler.jpg'):
        # Ground truth by Euler
        lorenz = Lorenz(self.params, self.initial_states, delta_t=self.dt)
        self.gt_Euler = list()
        for i in range(len(self.t)):
            x = lorenz.run()
            self.gt_Euler.append([x[-3], x[-2], x[-1]])
        self.gt_Euler = np.array(self.gt_Euler)
        fig = plt.figure(figsize=(12, 6))
        ax = fig.gca(projection='3d')  # 获取当前子图，指定三维模式
        ax.plot(self.gt_Euler[:, 0], self.gt_Euler[:, 1], self.gt_Euler[:, 2], lw=1.0, color='b')  # 画轨迹1
        plt.title('Lorenz system,$x_0$={},$y_0$={},$z_0$={}'.format(
                self.initial_states[0], self.initial_states[1], self.initial_states[2]))
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.show()
        plt.savefig(os.path.join(self.figure_dir, save_name))
        plt.close()

    def save_gt(self, method='Euler'):
        # save data
        assert method in ['odeint', 'Euler']
        if method == 'Euler':
            data = self.gt_Euler
        else:
            data = self.gt_ode
        sample_state = []
        save_path = os.path.join(self.data_dir, 'gt_{}_state.npy'.format(method))
        for i in range(self.gt_Euler.shape[0]):
            if i % self.h == 0:
                sample_state.append(data[i, :])
        sample_state = np.array(sample_state)
        np.save(save_path, sample_state)

    def GT_noise_from_Euler(self):
        initial_states = np.array(self.initial_states)
        states = initial_states #+ self.Q * np.ones(self.x_dim) #np.diagonal(np.eye()) #system noise
        measures = hx_x(states) + self.R * np.ones(self.z_dim)
        self.states_list.append(states)
        self.measures_list.append(measures)
        self.states_clean_list.append(initial_states)
        for i in range(self.epoch - 1):
            states_clean = lorenz63(states, self.dt)
            states = states_clean +  np.sqrt(self.Q) * np.random.normal(size=self.x_dim)
            measures = hx_x(states) + np.sqrt(self.R) * np.random.normal(size=self.z_dim)
            self.states_list.append(states)
            self.measures_list.append(measures)
            self.states_clean_list.append(states_clean)
            self.states_params_list.append(np.hstack((states, np.array(self.params))))
            self.states_params_clean_list.append(np.hstack((states_clean, np.array(self.params))))
        np.save(os.path.join(self.data_dir, 'states_clean.npy'), np.array(self.states_clean_list))
        np.save(os.path.join(self.data_dir, 'states.npy'), np.array(self.states_list))
        np.save(os.path.join(self.data_dir, 'states_params_clean.npy'), np.array(self.states_params_clean_list))
        np.save(os.path.join(self.data_dir, 'states_params.npy'), np.array(self.states_params_list))
        np.save(os.path.join(self.data_dir, 'measures.npy'), np.array(self.measures_list))

    def show_measures_from_Euler(self):
        measures_clean_list = (np.array(self.states_clean_list)[:, 0]).reshape(-1) # without system error and measurement error
        measures_system_error_list = (np.array(self.states_list)[:, 0]).reshape(-1) # with system error but no measurement error
        measures_list = (np.array(self.measures_list)).reshape(-1)
        colors = ['b', 'r', 'y']
        for value, color in zip([measures_clean_list, measures_system_error_list, measures_list], colors):
            plt.plot([i for i in range(self.epoch)], value, color)
        plt.xlabel('step')
        plt.ylabel('x (measurement)')
        plt.title('measures of Lorenz63')
        plt.legend(['without noise', 'with system noise', 'with system and measurement noise'])
        # plt.show()
        plt.savefig(os.path.join(self.figure_dir, 'measurement.jpg'))

    def show_states_from_Euler(self):
        names = ['states', 'states_clean']
        for value, name in zip([np.array(self.states_list),np.array(self.states_clean_list)], names):
            fig = plt.figure(figsize=(12, 6))
            ax = fig.gca(projection='3d')  # 获取当前子图，指定三维模式
            ax.plot(value[:, 0], value[:, 1], value[:, 2], lw=1.0, color='b')  # 画轨迹1
            plt.title('{} of Lorenz system, $x_0$={},$y_0$={},$z_0$={}'.format(
                    name, self.initial_states[0], self.initial_states[1], self.initial_states[2]))
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.show()
            plt.savefig(os.path.join(self.figure_dir, 'Lorenz63_{}_.jpg'.format(name)))
            plt.close()
    def show_observation_x(self):
        names = ['states', 'states_clean']
        colors = ['r', 'b']
        for value, name, color in zip([np.array(self.states_list),np.array(self.states_clean_list)], names, colors):
            plt.plot(value[:, 0], lw=1.0, color=color)  # 画轨迹1
            plt.title('{} of Lorenz system, $x_0$={},$y_0$={},$z_0$={}'.format(
                    name, self.initial_states[0], self.initial_states[1], self.initial_states[2]))
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.show()
        plt.savefig(os.path.join(self.figure_dir, 'Lorenz63_x.jpg'))
        plt.close()


def main():
    config = Config()
    initial_states = [1., 3., 5.]
    stage = 'train'
    if stage == 'train':
        train_nums = 6000
        for i in tqdm(range(train_nums)):
            print('------------{}---------'.format(i))
            config.initial_states = initial_states +  np.random.random(3) * 0.1
            generate_Lorenz63_data = Generate_Lorenz63_Data(config)
            generate_Lorenz63_data.initial_config_and_dirs(i)
            generate_Lorenz63_data.GT_noise_from_Euler()
    elif stage == 'test':
        # config.initial_states += np.random.random(3) * 4
        generate_Lorenz63_data = Generate_Lorenz63_Data(config)
        generate_Lorenz63_data.initial_config_and_dirs('test')
        generate_Lorenz63_data.GT_from_odeint()
        generate_Lorenz63_data.GT_from_Euler()
        generate_Lorenz63_data.save_gt(method='Euler')
        generate_Lorenz63_data.save_gt(method='odeint')
        generate_Lorenz63_data.GT_noise_from_Euler()
        generate_Lorenz63_data.show_measures_from_Euler()
        generate_Lorenz63_data.show_states_from_Euler()
        generate_Lorenz63_data.show_observation_x()

if __name__ == '__main__':
        main()
