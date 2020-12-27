from EnKF import EnsembleKalmanFilter
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
from config import Config
from utils import lorenz63, hx_x
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_value, mkdir

class EnKF_Lorenz():
    def __init__(self, data_name, d=5, enkf_index=1):
        self.data_dir = os.path.join('../Lorenz_data/' + data_name, 'data')
        self.figure_dir = os.path.join('../results/Lorenz/figures/EnKF', os.path.join(data_name, str(enkf_index)))
        self.logs_dir = os.path.join('../results/Lorenz/logs/EnKF', os.path.join(data_name, str(enkf_index)))
        self.detail_dir = os.path.join('../results/Lorenz/detail/EnKF', os.path.join(data_name, str(enkf_index)))
        self.config = Config()
        self.d = d
        self.x = self.config.initial_states_filter
        self.P = self.config.P
        self.dim_z = self.config.z_dim
        self.dt = self.config.dt
        self.N = self.config.N
        self.epoch = self.config.epoch
        self.Q = self.config.Q
        self.error3_all = list()
        self.error3_1_all = list()
        self.error4_all = list()
        self.error4_1_all = list()
        self.error5_all = list()
        self.error5_1_all = list()
        self.error6_all = list()
        self.error6_1_all = list()
        self.error7_all = list()
        self.error7_1_all = list()
        self.error8_all = list()
        self.error8_1_all = list()

    def init_load_data(self):
        mkdir(self.figure_dir)
        mkdir(self.logs_dir)
        mkdir(self.detail_dir)
        self.measurements = load_value(os.path.join(self.data_dir, 'measures.npy'))
        self.states = load_value(os.path.join(self.data_dir, 'states.npy'))
        self.states_clean = load_value(os.path.join(self.data_dir, 'states_clean.npy'))

    def run_enkf(self, index, steps):
        self.index = index
        assert steps <= self.epoch
        self.steps = steps
        enkf = EnsembleKalmanFilter(self.x, self.P, self.dim_z, self.dt, self.N, hx_x, lorenz63)
        self.post_states = list()
        self.prior_states = list()
        self.post_x = list()
        self.prior_x = list()
        for i in tqdm(range(steps)):
            enkf.predict(self.config.Q)
            self.prior_states.append(enkf.x_prior)
            self.prior_x.append(enkf.x_prior[0])
            enkf.update(self.measurements[i], self.config.R)
            self.post_states.append(enkf.x)
            self.post_x.append(enkf.x[0])

    def plot_enkf(self):
        # plot x
        colors = ['b', 'r', 'grey', 'y', 'k' ]
        for color, value in zip(colors, [self.states_clean[:,0], self.states[:,0],  self.post_x, self.prior_x, self.measurements]):
            plt.plot([i for i in range(len(self.states_clean))][self.d:self.steps], value[self.d:self.steps], color)
        plt.title('EnKF x Lorenz')
        plt.xlabel('step')
        plt.ylabel('x')
        plt.legend(['states without noise', 'states', 'EnKF post x', 'EnKF prior x', 'measurements'])
        plt.savefig(os.path.join(self.figure_dir, 'EnKF_x_results_{}.jpg'.format(self.index)))
        # plt.show()
        plt.close()


        # plot y,z
        names = ['y', 'z']
        for states_i in [1,2]:
            for index, value in enumerate([self.states_clean[:,states_i], self.states[:,states_i], np.array(self.post_states)[:, states_i], np.array(self.prior_states)[:, states_i]]):
                plt.plot([i for i in range(len(self.states_clean))][self.d: self.steps], value[self.d:self.steps], colors[index])
            plt.title('EnKF x Lorenz')
            plt.xlabel('step')
            plt.ylabel(names[states_i-1])
            plt.legend(['states without noise', 'states', 'EnKF post x', 'EnKF prior x'])
            plt.savefig(os.path.join(self.figure_dir, 'EnKF_{}_results_{}.jpg'.format(names[states_i-1], self.index)))
            # plt.show()
            plt.close()

    def save_details(self):
        pd.DataFrame(self.prior_states).to_csv(os.path.join(self.detail_dir, 'prior_states_{}.txt'.format(self.index)), mode='a+') # save prior_states
        pd.DataFrame(self.post_states).to_csv(os.path.join(self.detail_dir, 'post_states_{}.txt'.format(self.index)), mode='a+') # save post_states
        pd.DataFrame(self.prior_x).to_csv(os.path.join(self.detail_dir, 'prior_x_{}.txt'.format(self.index)), mode='a+') # save post_states
        pd.DataFrame(self.post_x).to_csv(os.path.join(self.detail_dir, 'post_x_{}.txt'.format(self.index)), mode='a+') # save post_states

    def print_save_error(self):
        error_1 = sqrt(mean_squared_error(self.states_clean[self.d:self.steps,0], self.states[self.d:self.steps,0]))
        error_2 = sqrt(mean_squared_error(self.states_clean[self.d:self.steps,0], self.measurements[self.d:self.steps]))
        error_3 = sqrt(mean_squared_error(self.states_clean[self.d:self.steps, 0], self.post_x[self.d:]))
        error_4 = sqrt(mean_squared_error(self.states[self.d:self.steps, 0], self.post_x[self.d:]))
        error_3_1 = sqrt(mean_squared_error(self.states_clean[self.d:self.steps, 0], self.prior_x[self.d:]))
        error_4_1 = sqrt(mean_squared_error(self.states[self.d:self.steps, 0], self.prior_x[self.d:]))
        error_5 = sqrt(mean_squared_error(self.states[self.d:self.steps, 1], np.array(self.post_states)[self.d:,1]))
        error_5_1 = sqrt(mean_squared_error(self.states[self.d:self.steps, 1], np.array(self.prior_states)[self.d:,1]))
        error_6 = sqrt(mean_squared_error(self.states_clean[self.d:self.steps, 1], np.array(self.post_states)[self.d:, 1]))
        error_6_1 = sqrt(mean_squared_error(self.states_clean[self.d:self.steps, 1], np.array(self.prior_states)[self.d:, 1]))
        error_7 = sqrt(mean_squared_error(self.states[self.d:self.steps, 2], np.array(self.prior_states)[self.d:,2]))
        error_7_1 = sqrt(mean_squared_error(self.states[self.d:self.steps, 2], np.array(self.prior_states)[self.d:,2]))
        error_8 = sqrt(mean_squared_error(self.states_clean[self.d:self.steps, 2], np.array(self.prior_states)[self.d:, 2]))
        error_8_1 = sqrt(mean_squared_error(self.states_clean[self.d:self.steps, 2], np.array(self.prior_states)[self.d:, 2]))
        print('-----------------{}----------------------'.format(self.index))
        print('noise x vs clean x: ', error_1)
        print('measurements and states_clean: ', error_2)
        print('post_x and clean x: ', error_3)
        print('prior_x and clean x: ', error_3_1)
        print('post_x and noisy x: ', error_4)
        print('prior_x and nosiy x: ', error_4_1)
        print('post_y and nosiy y: ', error_5)
        print('prior_y and nosiy y: ', error_5_1)
        print('post_y and clean y: ', error_6)
        print('prior_y and clean y: ', error_6_1)
        print('post_z and noisy z: ', error_7)
        print('prior_z and noisy z: ', error_7_1)
        print('post_z and clean z: ', error_8)
        print('prior_z and clean z: ', error_8_1)

        # f = open(os.path.join(self.logs_dir, 'logs_{}.txt'.format(self.index)), mode='a+')
        # f.writelines('noise x vs clean x:  {}\n'.format(error_1))
        # f.writelines('measurements and states_clean:  {}\n'.format(error_2))
        # f.writelines('post_x and clean x:  {}\n'.format(error_3))
        # f.writelines('prior_x and clean x:  {}\n'.format(error_3_1))
        # f.writelines('post_x and noisy x:  {}\n'.format(error_4))
        # f.writelines('prior_x and nosiy x:  {}\n'.format(error_4_1))
        # f.writelines('post_y and nosiy y:  {}\n'.format(error_5))
        # f.writelines('prior_y and nosiy y:  {}\n'.format(error_5_1))
        # f.writelines('post_y and clean y:  {}\n'.format(error_6))
        # f.writelines('prior_y and clean y:  {}\n'.format(error_6_1))
        # f.writelines('post_z and noisy z:  {}\n'.format(error_7))
        # f.writelines('prior_z and noisy z:  {}\n'.format(error_7_1))
        # f.writelines('post_z and clean z:  {}\n'.format(error_8))
        # f.writelines('prior_z and clean z:  {}\n'.format(error_8_1))
        # f.close()
        #
        # self.error3_all.append(error_3)
        # self.error3_1_all.append(error_3_1)
        # self.error4_all.append(error_4)
        # self.error4_1_all.append(error_4_1)
        # self.error5_all.append(error_5)
        # self.error5_1_all.append(error_5_1)
        # self.error6_all.append(error_6)
        # self.error6_1_all.append(error_6_1)
        # self.error7_all.append(error_7)
        # self.error7_1_all.append(error_7_1)
        # self.error8_all.append(error_8)
        # self.error8_1_all.append(error_8_1)


    def error_summary(self):
        names = ['post_x and clean x', 'prior_x and clean x', 'post_x and noisy x', 'prior_x and nosiy x',
                 'post_y and nosiy y', 'prior_y and nosiy y', 'post_y and clean y', 'prior_y and clean y',
                 'post_z and noisy z', 'prior_z and noisy z', 'post_z and clean z', 'prior_z and clean z']
        values = [self.error3_all, self.error3_1_all, self.error4_all, self.error4_1_all,
                  self.error5_all, self.error5_1_all, self.error6_all, self.error6_1_all,
                  self.error7_all, self.error7_1_all, self.error8_all, self.error8_1_all]
        f = open(os.path.join(self.logs_dir, 'error_summary.txt'), mode='a+')
        for name, value in zip(names, values):
            print('{},index={},steps={}, mean error={}, std error={}, top1 error={}'.format(
                name, self.index+1, self.steps, np.mean(value), np.std(value), np.min(value)))
            f.writelines('{},index={}, steps={}, mean error={}, std error={}, top1 error={}\n'.format(
                name, self.index+1, self.steps, np.mean(value), np.std(value), np.min(value)))
        f.close()

if __name__ == '__main__':
    enKF_lorenz = EnKF_Lorenz('test')
    enKF_lorenz.init_load_data()
    enKF_lorenz.run_enkf(0, 30)
    enKF_lorenz.plot_enkf()
    enKF_lorenz.print_save_error()
    # for i in range(100):
    #     enKF_lorenz.run_enkf(i, 1000)
    #     enKF_lorenz.plot_enkf()
    #     enKF_lorenz.save_details()
    #     enKF_lorenz.print_save_error()
    # enKF_lorenz.error_summary()

# -----------------0----------------------
# noise x vs clean x:  0.3285040475451077
# measurements and states_clean:  0.9570262669714023
# post_x and clean x:  0.37644432833714575
# prior_x and clean x:  0.40827725016623434
# post_x and noisy x:  0.40089161098936765
# prior_x and nosiy x:  0.4958626378164687
# post_y and nosiy y:  1.0039498613009368
# prior_y and nosiy y:  1.0218881677738494
# post_y and clean y:  0.9787971115875945
# prior_y and clean y:  0.9950731760990603
# post_z and noisy z:  1.1706176934626666
# prior_z and noisy z:  1.1706176934626666
# post_z and clean z:  1.1258411633400858
# prior_z and clean z:  1.1258411633400858