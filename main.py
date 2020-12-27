from Takens_EnKF_Lorenz import Takens_EnKF_Lorenz
from EnKF_Lorenz import EnKF_Lorenz
from generate_Lorenz_data import Generate_Lorenz63_Data
from config import Config
from tqdm import tqdm
import numpy as np

def generate_data():
    config = Config()
    initial_states = [1., 3., 5.]
    # stage = 'train'
    # if stage == 'train':
    train_nums = 6000
    for i in tqdm(range(train_nums)):
        print('------------{}---------'.format(i))
        config.initial_states = initial_states + np.random.random(3) * 4
        generate_Lorenz63_data = Generate_Lorenz63_Data(config)
        generate_Lorenz63_data.initial_config_and_dirs(i)
        generate_Lorenz63_data.GT_noise_from_Euler()
    # elif stage == 'test':
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

def do_EnKF(epoch):
    enKF_lorenz = EnKF_Lorenz('test')
    enKF_lorenz.init_load_data()
    enKF_lorenz.run_enkf(0, epoch)
    enKF_lorenz.plot_enkf()
    enKF_lorenz.print_save_error()

def do_Takens_EnKF(epoch):
    enKF_lorenz = Takens_EnKF_Lorenz('test')
    enKF_lorenz.init_load_data()
    enKF_lorenz.run_takens_enkf(0, epoch)
    enKF_lorenz.save_details()
    enKF_lorenz.plot_enkf()
    enKF_lorenz.print_save_error()

def main():
    #test 1
    epoch = 500
    generate_data()
    do_EnKF(epoch)
    do_Takens_EnKF(epoch)





