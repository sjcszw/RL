import math
import random
import sys
 
sys.path.append('../')

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from utilities.buffer import ReplayMemory,Transition

class Trainer(object):
    """Run trainning for given agent. 
    Optionally will visualise and save the results"""
    def __init__(self, config, agent):
        self.config = config
        self.agent = agent
        
        # define current model and target model
        num_input = config.hyperparameters["num_input"]
        num_output = config.hyperparameters["num_output"]
        num_u = config.hyperparameters["num_u"]
        self.cuda = config.use_GPU
        device = torch.device("cuda:0" if self.cuda else "cpu")
        self.current_net = agent(num_input=num_input, num_output=
                           num_output, num_u=num_u, cuda=self.cuda)
        self.current_net = self.current_net.to(device)
        self.target_net = agent(num_input=2, num_output=1, num_u=5,cuda=self.cuda).to(device)
        self.current_net = self.current_net.to(device)
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()
        
        # environments and buffer
        buffer_size = config.hyperparameters["buffer_size"]
        self.env = config.environment
        self.memory = ReplayMemory(buffer_size)
        self.batch_size = config.hyperparameters["batch_size"]
        
        # variables for training
        self.num_episodes = config.num_episodes_to_run
        self.target_update = config.hyperparameters["target_update"]
        self.gamma = config.hyperparameters["gamma"]
        self.loss_fun = torch.nn.MSELoss()  # Initializes the loss function
        learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.current_net.parameters(),
                                          lr=learning_rate)

        self.losses = []
        self.trac_time = []
        self.trac_reward = []
        self.step_total = 0  # records total steps for all trajectories
        self.TD_error = [] # records target difference error       
        
    def run(self):
        """Run the agent"""
        for epoch in range(self.num_episodes):
            self.run_agent_in_one_episode(epoch)
        self.env.close() 
            
    def run_agent_in_one_episode(self, epoch):
        state = self.env.reset()
        done = False  # records whether one trajectory is finished
        t_traj = 0  # records the number of steps in one trajectory
        r_traj = 0  # records the total rewards in one trajectory

        while not done:  # Second loop: within one tractory

#             position = str(env.state[0].round(decimals=2))
#             velocity = str(env.state[1].round(decimals=2))
            text = 'trajectory: '+str(epoch+1)
            self.env.render(text)  # visualization of the cart position 
            epsilon = greedy_epsilon(self.step_total)
            action = self.current_net.act(state, epsilon, self.env)

            if action[0]>10.0 or action[0]<-10.0:  # if infeasible, start a new trajectory
                break

            next_state, reward, done, _ = self.env.step(action,t_traj)

            # if done, the cart may go out of contraints, we don't 
            # want this fake reward to be saved.
    #         if done:
    #             break

            # save the step in the memory
            # transfer into type tensor
            state_      = self.vari_gpu(torch.FloatTensor(state)).unsqueeze(0)
            next_state_ = self.vari_gpu(torch.FloatTensor(next_state)).unsqueeze(0)
            action_     = self.vari_gpu(torch.FloatTensor(action)).unsqueeze(0)
            reward_     = self.vari_gpu(torch.FloatTensor([reward])).unsqueeze(0)
            done_       = self.vari_gpu(torch.FloatTensor([done])).unsqueeze(0)

            self.memory.push(state_, action_, next_state_, reward_, done_)
            state = next_state
            t_traj += 1
            r_traj += reward


            # Third loop: train the model
            loss = 0.0
            if self.step_total>self.batch_size:
                for k in range(self.config.num_taining_step_every_trajectory_step):
                    loss += self.train()
                    self.losses.append(loss/(k+1))

            # update the target network
            self.step_total += 1
            if self.step_total % self.target_update == 0:  
                self.target_net.load_state_dict(self.current_net.state_dict())

            if (self.step_total)%100 == 0 and self.step_total>128:
                print('[step_total: %d] training loss: %.3f' %
                                  (self.step_total, self.losses[self.step_total]))

            # compute the target difference error
            q_value, _ = self.current_net(state_,action_)  # Q(x0,u0)
            v_value, _ = self.target_net(next_state_)  # V(x1)
            q_value = q_value.data[0,0].cpu().numpy()
            v_value = v_value.data[0,0].cpu().numpy()
            self.TD_error.append(reward + self.gamma * v_value * (1-done) - q_value)

        #  record data of this trajectory in lists
        self.trac_time.append(t_traj)
        self.trac_reward.append(r_traj)

       
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state = torch.cat(batch.state_)
        action = torch.cat(batch.action_)
        next_state = torch.cat(batch.next_state_)
        reward = torch.cat(batch.reward_)
        done = torch.cat(batch.done_)

        q_value, _ = self.current_net(state,action)  # Q(x0,u0)
        v_value, _ = self.target_net(next_state)  # V(x1)

        # compute the expected Q values
        expected_q_value = reward + self.gamma * v_value * (1-done)

        # compute loss
        loss = self.loss_fun(expected_q_value, q_value) 

        # optimize the model  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data
    
    def vari_gpu(self, var):
        if self.cuda:
            var = var.cuda()
        return var

# Epsilon-greedy exploration
# The epsilon decreases exponetially as time goes by.
def greedy_epsilon(step_total):
    epsilon_start = 0.8
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon = epsilon_final + (epsilon_start-epsilon_final) * math.exp(-1.*step_total/epsilon_decay)
    return epsilon    