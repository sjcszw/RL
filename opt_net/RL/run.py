import sys
 
sys.path.append('../')

from agents.Q_learning_off_policy import Q_learning_off_policy
from agents.trainer import Trainer
from environments.cart import MyEnv
from utilities.model_operation import save_model
from utilities.config import Config




config = Config()
config.seed = 1
config.environment = MyEnv()
config.num_episodes_to_run = 10
config.num_taining_step_every_trajectory_step = 3
config.visualise_results = True
config.file_to_save_data_results = './model/checkpoint.pth.tar'
config.file_to_save_results_graph = None
config.use_GPU = False
config.save_model = False


config = Config()
config.seed = 1
config.environment = MyEnv()
config.num_episodes_to_run = 10
config.num_taining_step_every_trajectory_step = 3
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
}

if __name__ == "__main__":
    AGENT = Q_learning_off_policy
    trainer = Trainer(config, AGENT)
    trainer.run()

    if config.file_to_save_data_results:
        save_model(trainer,config.file_to_save_data_results)