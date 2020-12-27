import numpy as np
from utils import convert_w


class Config():
    def __init__(self):
        """generate Lorenz data"""
        self.params = [10., 28., 8. / 3.]
        self.initial_states = [1., 3., 5.]
        self.dt = 0.01
        self.end = 6
        self.epoch = int(self.end / self.dt) # generate a long data


        self.z_dim = 1
        self.x_dim = 3
        self.Q = 0.4
        self.R = 0.4
        self.h = 1  # EnKF step


        self.initial_states_filter = np.array([1.5, 2, 6])
        self.P = np.ones((self.x_dim, self.x_dim)) * 0.1
        self.N = 50

    def update_config_enkf(self, config):
        self.initial_states_filter = config['initial_states_filter']
        self.N = config['N']

    def update_config_generate(self, config):
        self.dt = config['dt']
        self.Q = config['Q']
        self.R = config['R']
def main():
    config = Config()
    print(config.__dict__)

if __name__ == '__main__':
    main()