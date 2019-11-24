import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import MultivariateNormal


class Policy_Gradient(object):
    """Policy gradient trainning. 
    """
    def __init__(self, config, net):
        self.config = config
        self.net = net  # NN for estimation of policy
        
        # define poliy net
        num_input = config.hyperparameters["num_input"]
        num_output = config.hyperparameters["num_output"]
        num_u = config.hyperparameters["num_u"]
        self.cuda = config.use_GPU
        device = torch.device("cuda:0" if self.cuda else "cpu")
        self.policy_net = net(num_input=num_input, num_output=
                           num_output, num_u=num_u, cuda=self.cuda)
        self.policy_net = self.policy_net.to(device)
        
        # environments and buffer
        self.env = config.environment
        self.batch_size = config.hyperparameters["batch_size"]
        self.stds = config.hyperparameters["action_stds"]
        
        # variables for training
        self.num_epoch = config.num_epoch_to_run
        self.target_update = config.hyperparameters["target_update"]
        self.gamma = config.hyperparameters["gamma"]
        self.loss_fun = torch.nn.MSELoss()  # Initializes the loss function
        learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)

        self.losses = []
        self.trac_time = []
        self.trac_reward = []
        self.step_total = 0  # records total steps for all trajectories      
        
    def run(self):
        """Run the agent"""
        for epoch in range(self.num_epoch):  # First loop
            self.run_agent_in_one_epoch(epoch)
        self.env.close() 
            
    def run_agent_in_one_epoch(self, epoch):
        """One epoch stops if number pf steps reach batchsize.
        Then trains the policy net based on this epoch.
        """
 
        # makes some empty lists for logging.
        batch_states = []       # for states
        batch_acts = []         # for actions
        batch_log_prob = []     # for log(prob) of actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # resets episode-specific variables
        state = self.env.reset()  # first states in one trajectory
        done = False            # records whether one trajectory is finished
        ep_rews = []            # records the total rewards in one trajectory

        text = 'epoch: '+str(epoch+1)
        
        # renders first episode of each epoch
        finished_rendering_this_epoch = False

        # collects experience by acting in the environment with current policy
        while True:

            # visualization of the cart position 
            if not(finished_rendering_this_epoch):
                self.env.render(text)  
            
            action, log_prob = self.act(state)
            action_ = (action.cpu().detach().numpy())
            next_state, reward, done, _ = self.env.step(action_)
            
            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # resets episode-specific variables
                self.env.close() 
                state, done, ep_rews = self.env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # ends experience loop if we have enough of it
                if len(batch_states) > self.batch_size:
                    break
            
            # only if the env is not done:
            # saves obs
            batch_states.append(state.copy())
            # saves action, reward
            batch_acts.append(action)
            batch_log_prob.append(log_prob)
            ep_rews.append(reward)
            
            # updates new state
            state = next_state


        # takes a single policy gradient update step
        num_traj =  len(batch_rets)
        batch_loss = self.train(num_traj,batch_log_prob, 
                                batch_weights)
  


        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch, batch_loss, np.mean(batch_rets),
                 np.mean(batch_lens)))

        #  record data of this epoch in lists
        self.losses.append(batch_loss)
        self.trac_time += batch_lens
        self.trac_reward += batch_rets
        
    def act(self, state):
        """The action excuted by gaussian policies
        state -> policy net -> action mean
        self.stds(standard deviation): manual setting
        """
        state   = Variable(torch.FloatTensor(state)).unsqueeze(0) # adds extra dim when single input
        state = self.vari_gpu(state)
        _, action_mean = self.policy_net(state)
        #print('act:model action ',action_mean)
        
        # builds distribution
        # if action is out of env action range, resample it
        high = self.env.action_space.high
        low = self.env.action_space.low 
        while True:
            action_distribution = MultivariateNormal(
                action_mean,torch.abs(self.stds))
            action =  action_distribution.sample()  # random action sampling
            if ((action.cpu().numpy() <= high) and 
                (action.cpu().numpy() >= low)):
                break
                
        # log probability of chosen action
        log_prob = action_distribution.log_prob(action).reshape(1)
        return action, log_prob


    def train(self,num_traj,batch_log_prob,batch_weights):
        '''takes a single policy gradient update step
        '''
        log_prob = torch.cat(batch_log_prob)
        weights = torch.FloatTensor(batch_weights)
        #print(batch_log_prob,batch_weights)
        # compute loss
        loss = (torch.sum(torch.mul(log_prob, weights),-1)
                /num_traj)

        # optimize the model  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data
    
    def vari_gpu(self, var):
        if self.cuda:
            var = var.cuda()
        return var 