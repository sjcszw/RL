import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from matrix_square_root import sqrtm

def QP_layer(nz, nineq):
    """Builds the QP layer.

    The optimization problem is of the form
        \hat z =   argmin_z z^T*Q*z + p^T*z
                subject to G*z <= h
    where Q \in S^{nz,nz},
        S^{nz,nz} is the set of all positive semi-definite matrices,
        p \in R^{nz}
        G \in R^{nineq,nz}
        h \in R^{nineq}
    
    Take the matrix square-root of Q：mentioned in paper P19
    (Differentiable Convex Optimization Layers).
    """
    Q_sqrt = cp.Parameter((nz, nz))
    p = cp.Parameter(nz)
    G = cp.Parameter((nineq, nz))
    h = cp.Parameter(nineq)
    z = cp.Variable(nz)
    obj = cp.Minimize(cp.sum_squares(Q_sqrt*z) + p.T@z)
    cons = [ G@z <= h ]
    prob = cp. Problem(obj, cons)
    assert prob.is_dpp()

    layer = CvxpyLayer (prob, 
                        parameters =[Q_sqrt, p, G, h], 
                        variables =[ z ])
    return layer


class CvxNet(nn.Module):
    """Builds the strucre of the cvx net.
    
    The struture is x0-QP-[cost,u].
    QP layer:
        \hat z =   argmin_z z^T*Q*z + p^T*z
                subject to G*z <= h
    parameters: Q_sqrt, p, G, h
    """
    
    def __init__(self, num_input, num_output, num_u=5, eps=1e-4, cuda=False,collect=False):

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
        
        self.layer = QP_layer(nz=num_u,nineq=self.num_ineq)
        
        self.Q = Parameter(torch.rand(num_input, num_input),requires_grad=True)
        self.R = Parameter(torch.rand(num_output,num_output),requires_grad=True)
        
        self.A = Parameter(torch.rand(num_input, num_input),requires_grad=True)
        self.B = Parameter(torch.rand(num_input, num_output),requires_grad=True)

        self.s0 = Parameter(10*torch.ones(2*num_u),requires_grad=False)
        self.s1 = Parameter(4*torch.ones(num_input*self.N),requires_grad=False)
        self.s2 = Parameter(4*torch.ones(num_input*self.N),requires_grad=False)

        if collect==True:
            self.Q_sqrt = Parameter(torch.Tensor([[1.0,0.0],[0.0,1.0]]),requires_grad=True)
            self.R_sqrt = Parameter(torch.eye(num_output),requires_grad=True)
            self.A = Parameter(torch.Tensor([[1.0,1.0],[0,1.0]]),requires_grad=True)
            self.B = Parameter(torch.Tensor([[0.5],[1.0]]),requires_grad=True)
            self.s0 = Parameter(10*torch.ones(2*num_u),requires_grad=False)
            self.s1 = Parameter(4*torch.ones(num_input*self.N),requires_grad=False)
            self.s2 = Parameter(4*torch.ones(num_input*self.N),requires_grad=False)
                   
        weight = torch.zeros(num_u)
        weight[0] = 1.0
        self.weight = Parameter(weight,requires_grad=False)
            

    def forward(self, x):
        """Builds the forward strucre of the QPNet.
        Sequence: x0-QP-[cost,u].
        QP parameters: Q_sqrt, p, G, h
        """
        # input x0 and batch size 
        num_batch = x.size(0)
        x0 = x.view(num_batch, -1)
        
        A_hat = self.build_A_block()
        B_hat = self.build_B_block()
        
        # Q_sqrt in QP
        Q = self.Q.mm(self.Q.t())
        R = self.R.mm(self.R.t())
        R_diag = self.build_Rdiagnol_block(R)
        Q_hat, Q_diag = self.build_Q_block(Q, Q, R, B_hat)
        Q_sqrt_hat = sqrtm(Q_hat)  # computs sqrt of Q
        Q_sqrt_hat = Q_sqrt_hat.repeat(num_batch,1,1)  # builds batch
                
        # p in QP  p = 2 * (Q_diag*B_hat)^T * (A_hat*x0)
        A_x0 = A_hat.mm(x0.t()).t()  # presents[x1;x2;...;xN] size: batch * dim(x1;x2;...;xN)
        p = 2*A_x0.mm(Q_diag.mm(B_hat))
        
        # G in QP
        G = self.build_G_block(B_hat)
        G = G.repeat(num_batch,1,1)  # builds batch
        
        # h in QP
        s0 = self.s0.repeat(num_batch,1)  # builds batch
        s1 = self.s1.repeat(num_batch,1)  # builds batch
        s1 -= A_x0
        s2 = self.s2.repeat(num_batch,1)  # builds batch
        s2 += A_x0
        h = torch.cat((s0,s1,s2),1)
        
        
        
        # gets the solution of the basic optimization problem
        self.Qr = Q_sqrt_hat
        u_opt, = self.layer(Q_sqrt_hat, p, G, h)  # size: batch*dim(u)
        
        # get the optimal cost
        # a+b: sum(i:1 to N): xi^T*Q*xi + u(i-1)^T*R*u(i-1)
        # c: x0^T*Q*x0
        a = (u_opt.mm(Q_hat)*u_opt + p*u_opt).sum(1)
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
        """ (num_imput*N) x (num_imput*N)
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
        Q_hat = B_hat^T * diag(Q) * B_hat + diag(R)
        """
        Q_diag = self.build_Qdiagnol_block(Q,P)
        R_diag = self.build_Rdiagnol_block(R)
        Q_hat = B_hat.t().mm(Q_diag.mm(B_hat)) + R_diag
        return Q_hat,Q_diag 
        
        
    def build_G_block(self,B_hat):
        """Build the G matrix so that MPC is tranfered into basic optimization problem
        G = [eye(num_u)]
            [-eye(num_u)]
            [   B_hat  ]
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