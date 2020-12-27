import os
import numpy as np
import random

from utils import load_value


class Takens():
    def __init__(self, train_dir='../Lorenz_data', delay=4+1, train_num=6000, lockout_num=0, N=20):
        self.train_dir = train_dir
        self.delay = delay
        self.train_num = train_num
        self.lockout_num = lockout_num
        self.N = N

    # def get_sample_num(self):
    #     self.sample_num = np.ones(len(self.train_files)) * (self.train_num // len(self.train_files))
    #     self.sample_num[-1] += self.train_num - np.sum(self.sample_num)

    def get_train_data(self, k):
        self.measurements = list()
        self.states = list()
        self.next_measurement = list()
        self.next_states = list()
        for index in range(self.train_num):
            data_dir = os.path.join('../Lorenz_data/' + str(index), 'data')
            measurements = load_value(os.path.join(data_dir, 'measures.npy'))
            states = load_value(os.path.join(data_dir, 'states.npy'))
            # sample = random.sample(np.shape(measurements)[0]-self.delay, self.sample_num[index]) #从训练集文件中采样
            self.measurements.append(measurements[k:k-self.delay:-1])
            self.states.append(states[k:k-self.delay:-1,0])
            self.next_measurement.append(measurements[k+1].tolist()[0])
            self.next_states.append(states[k+1, :])

    def get_update_satates(self, input_array):
        assert np.shape(input_array)[0] == self.delay
        # assert len(self.measurements) == self.train_num
        # assert len(self.states) == self.train_num
        dist_list = list()
        for i in range(self.train_num):
            # dist = np.sum((input_array - self.states[i])**2)
            dist = np.sum((input_array - self.measurements[i])**2)
            dist_list.append(dist)
        argsort = np.argsort(dist_list)
        best_neighbors_index = argsort[self.lockout_num: self.lockout_num + 20]
        # print('best_neighbors_index: ', best_neighbors_index)
        best_pred_state = np.mean(np.array(self.next_states)[best_neighbors_index], axis=0)
        best_pred_measure = np.mean(np.array(self.next_measurement)[best_neighbors_index])
        return best_pred_state, best_pred_measure


def main():
    takens = Takens()
    takens.get_train_data(6)
    best_pred_state, best_pred_measure = takens.get_update_satates(np.array([1,2,3,4,5]))
    print(best_pred_state, best_pred_measure)


if __name__ == '__main__':
    main()


