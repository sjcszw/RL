import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from qpth.qp import QPFunction


class QpNet(nn.Module):
    """Builds the strucre of the QPNet.
    
    The struture is x0-QP-[cost,u].
    QP means the optimization problem layer over`nz` variables, 
    having `nineq` inequality constraints and `neq` equality 
    constraints.
    The optimization problem is of the form
        \hat z =   argmin_z 1/2*z^T*Q*z + p^T*z
                subject to G*z <= h
                           A*z = b
    where Q \in S^{nz,nz},
        S^{nz,nz} is the set of all positive semi-definite matrices,
        p \in R^{nz}
        G \in R^{nineq,nz}
        h \in R^{nineq}
        A \in R^{neq,nz}
        b \in R^{neq}
        This layer has Q = L*L^T+ϵ*I where L is a lower-triangular matrix,
        and h = G*z0 + s0, b = A*z0 for some learnable z0 and s0,  
        to ensure the problem is always feasible.
    """
    
    def __init__(self, num_input, num_output, num_u=5, eps=1e-4, cuda=True,collect=False):

        """Initiates OptNet."""
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.num_u = num_u
        self.eps = eps
        self.cuda = cuda
        self.collect = collect
        
        self.N = int(self.num_u/self.num_output)  # get the number of the finite steps in MPC
        self.num_ineq = 2*(num_u + num_input*self.N)
        
        self.Q = Parameter(torch.rand(num_input, num_input),requires_grad=True)
        self.R = Parameter(torch.rand(num_output,num_output),requires_grad=True)
        
        self.A = Parameter(torch.rand(num_input, num_input),requires_grad=True)
        self.B = Parameter(torch.rand(num_input, num_output),requires_grad=True)

        self.s0 = Parameter(10*torch.ones(1,2*num_u),requires_grad=False)
        self.s1 = Parameter(4*torch.ones(1,num_input*self.N),requires_grad=False)
        self.s2 = Parameter(4*torch.ones(1,num_input*self.N),requires_grad=False)

        if collect==True:
            self.Q = Parameter(torch.Tensor([[1.0,0.0],[0.0,1.0]]),requires_grad=True)
            self.R = Parameter(torch.eye(num_output),requires_grad=True)
            self.A = Parameter(torch.Tensor([[1.0,1.0],[0,1.0]]),requires_grad=True)
            self.B = Parameter(torch.Tensor([[0.5],[1.0]]),requires_grad=True)
            self.s0 = Parameter(10*torch.ones(1,2*num_u),requires_grad=False)
            self.s1 = Parameter(4*torch.ones(1,num_input*self.N),requires_grad=False)
            self.s2 = Parameter(4*torch.ones(1,num_input*self.N),requires_grad=False)
                   
        weight = torch.zeros(num_u)
        weight[0] = 1.0
        self.weight = Parameter(weight,requires_grad=False)
            

    def forward(self, x):
        """Builds the forward strucre of the QPNet.
        Sequence: x0-QP-[cost,u].
        """
        
        A_hat = self.build_A_block()
        B_hat = self.build_B_block()
        
       
        Q = self.Q.mm(self.Q.t())
        R = self.R.mm(self.R.t())
        R_diag = self.build_Rdiagnol_block(R)
        Q_hat, Q_diag = self.build_Q_block(Q, Q, R, B_hat) 
                
        # linear layer  p = 2 * (Q_diag*B_hat)^T * (A_hat*x0)
        num_batch = x.size(0)
        x0 = x.view(num_batch, -1)
        A_x0 = A_hat.mm(x0.t()).t()  # present[x1;x2;...;xN] size: batch * dim(x1;x2;...;xN)
        p = 2*A_x0.mm(Q_diag.mm(B_hat))
        
        G = self.build_G_block(B_hat)
        s0 = self.s0.repeat(num_batch,1)
        s1 = self.s1.repeat(num_batch,1)
        s1 -= A_x0
        s2 = self.s2.repeat(num_batch,1)
        s2 += A_x0
        h = torch.cat((s0,s1,s2),1)
        
        # gets the solution of the basic optimization problem
        e = Variable(torch.Tensor())
        self.Qr = Q_hat
        u_opt = QPFunction(verbose=-1)(Q_hat, p, G, h, e, e)  # size: batch*dim(u)
        # shows solver warning if verbose >= 0
        
        # get the optimal cost
        a = (u_opt.mm(Q_hat)*u_opt/2 + p*u_opt).sum(1)
        b = (A_x0.mm(Q_diag)*A_x0).sum(1)
        c = (x0.mm(Q)*x0).sum(1)

        cost_opt = (a+b+c).unsqueeze(1)  # size: batch*1
        # u0 = u_opt.mv(self.weight).unsqueeze(1)  # only the fisrt action
        cost_and_u = torch.cat((0.1*cost_opt,u_opt),1)
        
        return cost_and_u
    
    def build_A_block(self):
        """
        [A]
        [A^2] 
        [A^3]
        [...]
        """
        N = self.N  # number of MPC steps
        A = self.A
        
        row_list = [A]  # reocrd the every row in B_hat
        
        for i in range(1, N):
            A = A.mm(self.A)
            row_list.append(A)
        return torch.cat(row_list,0)
    
    def build_B_block(self):
        """In MPC, express x vector in u vector and compute the new big B_hat matrix
        [B 0 0 ...
        [AB B 0
        ...
        """

        N = self.N  # number of MPC steps
        row_list = []  # reocrd the every row in B_hat
        
        first_block = self.B
        zero = Variable(torch.zeros(self.num_input, self.num_output*(N-1)))
        zero = self.vari_gpu(zero)
        row= torch.cat([first_block, zero],1)
        row_list.append(row)
        
        for i in range(1, N):
            first_block = self.A.mm(first_block)
            row = torch.cat([first_block, row[:,:self.num_output*(N-1)]],1)
            row_list.append(row)  
            
        return torch.cat(row_list,0)
        
        
    def build_Qdiagnol_block(self, Q, P):
        """ (num_imput*(N-1)) x (num_imput*(N-1))
        The last block is P for x(N)"""
        
        N = self.N  # number of MPC steps
        num_input = self.num_input
        
        row_list = []  # reocrd the every row in B_hat
        zero = Variable(torch.zeros(num_input, num_input*(N-1)))
        zero = self.vari_gpu(zero)
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
        zero = Variable(torch.zeros(num_output, num_output*(N-1)))
        zero = self.vari_gpu(zero)
        row_long = torch.cat([zero, R, zero],1)  # [0 0 ... Q 0 0 ...]
        
        for i in range(N, 0, -1):
            row_list.append(row_long[:, (i-1)*num_output : (i+N-1)*num_output])
        return torch.cat(row_list,0)
        
    def build_Q_block(self, Q, P, R, B_hat):
        """Build the Q_hat matrix so that MPC is tranfered into basic optimization problem
        1/2*Q_hat = B_hat^T * diag(Q) * B_hat + diag(R)
        """
        
        Q_diag = self.build_Qdiagnol_block(Q,P)
        R_diag = self.build_Rdiagnol_block(R)
        Q_hat = B_hat.t().mm(Q_diag.mm(B_hat)) + R_diag
        return 2*Q_hat,Q_diag 
        
        
    def build_G_block(self,B_hat):
        """Build the G matrix so that MPC is tranfered into basic optimization problem
        G = [eye(num_u)]
            [   B_hat  ]
            [-eye(num_u)]
            [  -B_hat  ]
        """
        
        eye = Variable(torch.eye(self.num_u))
        eye = self.vari_gpu(eye)
        G = torch.cat((eye, -eye, B_hat, -B_hat), 0)
        # print(self.B_hat)
        # print(G.size())
        return G
    
    def vari_gpu(self, var):
        if self.cuda:
            var = var.cuda()
            
        return var