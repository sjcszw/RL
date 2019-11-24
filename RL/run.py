import sys
 
sys.path.append('../')

import torch

from agents.cvx_nets import Cvx_Nets
from agents.policy_gradient_agent import Policy_Gradient
from agents.Q_learning_agent import Q_Learning
from environments.cart import MyEnv
from utilities.config import Config


config = Config()
config.seed = 1
config.environment = MyEnv()

# for Q-learning
config.num_episodes_to_run = 10
config.num_taining_step_every_trajectory_step = 3

# for policy gradients
config.num_epoch_to_run = 1  

config.visualise_results = True
config.file_to_save_data_results = './save/model/checkpoint.pth.tar'
config.file_to_save_results_graph = None
config.use_GPU = False
config.save_model = False


config.hyperparameters = {
            "num_input": 2,
            "num_output": 1,
            "num_u": 5,
            "collect": False,
            "learning_rate": 1e-3,
            "linear_hidden_units": [30, 15],
            "gamma": 1.0,  # discount_rate
            "target_update": 10,
            "batch_size": 20,
            "buffer_size": 10000,
            "action_stds": torch.tensor([[0.1]]),  # standard deviation for Gaussian policy
}

if __name__ == "__main__":
    net = Cvx_Nets
    trainer = Q_Learning(config, net)
    trainer.run()

    if config.file_to_save_data_results:
        save_model(trainer,config.file_to_save_data_results)