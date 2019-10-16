import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qpth.qp import QPFunction

class Q_Online(nn.Module):
    """Builds the strucre of the dnn based on QpNet for DQN
    The struture is FC-ReLU-FC-ReLU-Qp-[Q_value, u_opt[0]].
    """
    
    def __init__(self, num_input, num_output, num_hidden, num_u=5, eps=1e-4):
        """Inits the class.
        num_input: dimension of state variable
        num_output: dimension of action variable
        num_u: num_output * horizon length
        num_hidden: dimension in hidden FC layer
        num_equ >= (# eq constaints) * horizon length
        """
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_u = num_u
        self.eps = eps
        
        self.N = int(self.num_u/self.num_output)  # get the number of the finite steps in MPC
        self.num_ineq = num_u + num_input*(self.N-1)
        
        self.normal_layers = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_u),
            nn.LeakyReLU(),
        )
        
        # builds baisc tensors for QP parameters
        if torch.cuda.is_available():  # cuda or cpu
            self.Variable = lambda *args, **kwargs: \
                torch.autograd.Variable(*args, **kwargs).cuda()  # Variable: not require gra
            self.Parameter = lambda *args, **kwargs: \
                torch.nn.parameter.Parameter(*args, **kwargs).cuda()  # Parameter: require gra
        else:
            self.Variable = lambda *args, **kwargs: \
                torch.autograd.Variable(*args, **kwargs)
            self.Parameter = lambda *args, **kwargs: \
                torch.nn.parameter.Parameter(*args, **kwargs)
        self.M = self.Variable(torch.tril(torch.ones(num_input, num_input)))
        self.L = self.Parameter(torch.tril(torch.rand(num_input, num_input)))
        self.L.retain_grad()
        self.M_P = self.Variable(torch.tril(torch.ones(num_input, num_input)))
        self.L_P = self.Parameter(torch.tril(torch.rand(num_input, num_input)))
        self.L_P.retain_grad()
        self.M_R = self.Variable(torch.tril(torch.ones(num_output, num_output)))
        self.L_R = self.Parameter(torch.tril(torch.rand(num_output, num_output)))
        self.L_R.retain_grad()
        self.I = self.Variable(torch.eye(num_input))
        self.I_R = self.Variable(torch.eye(num_output))
        self.A = self.Parameter(torch.Tensor(num_input,num_input).uniform_(-1,1))
        self.A.retain_grad()
        self.B = self.Parameter(torch.Tensor(num_input,num_output).uniform_(-1,1))
        self.B.retain_grad()
        self.u0 = self.Parameter(torch.zeros(num_u))
        self.u0.retain_grad()
        self.s0 = self.Parameter(torch.ones(self.num_ineq))
        self.s0.retain_grad()
        self.B_hat = self.build_B_block()
        
        # set up the QP parameters Q=L*L^T+ϵ*I, h = G*u_0+s_0
        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*self.I
        L_P = self.M_P*self.L_P
        P = L_P.mm(L_P.t()) + self.eps*self.I
        L_R = self.M_R*self.L_R
        R = L_R.mm(L_R.t()) + self.eps*self.I_R
        self.Q_hat = self.build_Q_block(Q, P, R)
        
        self.G = self.build_G_block()
        self.h = self.G.mv(self.u0)+self.s0
        
        weight = torch.zeros(num_u)
        weight[0] = 1.0
        self.weight = self.Variable(weight)
                
    def forward(self, x):
        """Builds the forward strucre of the QPNet.
        Sequence: FC-ReLU-(BN)-FC-ReLU-(BN)-QP-[Q_value, u_opt[0]].
        """
        
        x = self.normal_layers(x)

        # gets the solution of the basic optimization problem
        e = self.Variable(torch.Tensor())
        u_opt = QPFunction(verbose=-1)(self.Q_hat, x, self.G, self.h, e, e)
        u0 = u_opt.mv(self.weight)
        Q_value = (u_opt.mm(self.Q_hat)*u_opt/2 + x*u_opt).sum(1).unsqueeze(1)
        
        return Q_value, u0

    def build_B_block(self):
        """In MPC, express x vector in u vector and compute the new big B_hat matrix
        [B 0 0 ...
        [AB B 0
        ...
        """

        N = self.N  # number of MPC steps
        row_list = []  # reocrd the every row in B_hat
        
        first_block = self.B
        zero = self.Variable(torch.zeros(self.num_input, self.num_output*(N-1)))
        row= torch.cat([first_block, zero],1)
        row_list.append(row)
        
        for i in range(1, N-1):
            first_block = self.A.mm(first_block)
            row = torch.cat([first_block, row[:,:self.num_output*(N-1)]],1)
            row_list.append(row)  
            
        return torch.cat(row_list,0)
        
        
    def build_Qdiagnol_block(self, Q, P):
        """ (num_imput*(N-1)) x (num_imput*(N-1))
        The last block is P for x(N)"""
        
        N = self.N-1  # number of MPC steps
        num_input = self.num_input
        
        row_list = []  # reocrd the every row in B_hat
        zero = self.Variable(torch.zeros(num_input, num_input*(N-1)))
        row_long = torch.cat([zero, Q, zero],1)  # [0 0 ... Q 0 0 ...]
        for i in range(N, 1, -1):
            row_list.append(row_long[:, (i-1)*num_input : (i+N-1)*num_input])
            
        row = torch.cat([zero, P],1)  # last line by [0 P]
        row_list.append(row)
        
        return torch.cat(row_list,0)
    
    def build_Rdiagnol_block(self, R):
        """
        [R 0 0 ...
        [0 R 0
        ...
        """
        N = self.N  # number of MPC steps
        num_output = self.num_output
        
        row_list = []  # reocrd the every row in B_hat
        zero = self.Variable(torch.zeros(num_output, num_output*(N-1)))
        
        row_long = torch.cat([zero, R, zero],1)  # [0 0 ... Q 0 0 ...]
        
        for i in range(N, 0, -1):
            row_list.append(row_long[:, (i-1)*num_output : (i+N-1)*num_output])
        return torch.cat(row_list,0)
        
    def build_Q_block(self, Q, P, R):
        """Build the Q_hat matrix so that MPC is tranfered into basic optimization problem
        Q_hat = B_hat^T * diag(Q) * B_hat + diag(R)
        """
        
        Q_diag = self.build_Qdiagnol_block(Q,P)
        R_diag = self.build_Rdiagnol_block(R)
        Q_hat = self.B_hat.t().mm(Q_diag.mm(self.B_hat)) + R_diag
        return Q_hat 
        
        
    def build_G_block(self):
        """Build the G matrix so that MPC is tranfered into basic optimization problem
        G = [eye(num_u)]
            [   B_hat  ]
        """
        
        eye = self.Variable(torch.eye(self.num_u))
        G = torch.cat((eye, self.B_hat), 0)
        # print(self.B_hat)
        # print(G.size())
        return G

    def act(self, state, epsilon, env):
        """The action excuted by epsilon-greedy exploration
        """
        if random.random() > epsilon:
            state   = self.Variable(torch.FloatTensor(state)).unsqueeze(0) # adds extra dim when single input
            _, u_opt = self.forward(state)
            action = (u_opt.cpu().detach().numpy())  # compute the u*[0] 
            #print('act:q_value ',q_value)
            #print('act:model action ',action)
        else:
            rand = np.random.rand(int(np.array(env.action_space.shape)))
            high = env.action_space.high
            low = env.action_space.low
            action = low + rand*(high-low)
            #print('act: ',action)
        return action
