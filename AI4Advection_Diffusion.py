import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sympy as sp
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")

bias_initalizer = torch.tensor([0], dtype=torch.float32)

class PDE_Solver(nn.Module):
    def __init__(self,eq,u,dx,dy,dt,nx,ny,solver,omega=1,s = 0):
        super(PDE_Solver, self).__init__()
        self.s = s
        A =solver(eq,u,dt,dx,dy)
        A = torch.from_numpy(A).unsqueeze(0).unsqueeze(0).float()
        self.init_condition = torch.tensor(u.data[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.nx = nx
        self.u = u
        self.ny = ny
        self.omega = omega
        self.nlevel = int(math.log(min(nx, ny), 2)) + 1 
        A_res = torch.zeros((1,1,2,2))
        A_res[0,0,:,:] = 0.25
        self.diag = np.array(A)[0,0,1,1]
        A[0,0,1,1] = 0
        self.smooth = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=0)
        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)
        self.smooth.weight.data = A
        self.res.weight.data = A_res
        self.A = A

        self.smooth.bias.data = bias_initalizer
        self.res.bias.data = bias_initalizer
        
    def boundary_condition(self,u):
        u_pdd = torch.zeros((1,1, self.ny+2,self.nx+2), dtype=torch.float32)
        u_pdd[0,0,1:-1,1:-1] = u[0,0,:,:]
        u_pdd[0,0,-1,1:-1] = u[0,0,-1,:]
        u_pdd[0,0,0,1:-1] = u[0,0,0,:]
        u_pdd[0,0,1:-1,-1] = u[0,0,:,-1]
        u_pdd[0,0,1:-1,0] = u[0,0,:,0]
        return u_pdd
    
    def F_cycle(self, iteration):
        """
        b is Right-end item (source item) of advection equation
        """
        b=0.0
        u = self.init_condition.clone()
        for _ in range(iteration):
            r = self.smooth(self.boundary_condition(u))-b
            r_s = []
            r_s.append(r)
            e_coarse = torch.zeros((1,1,1,1),device = device)
            for i in range(1, self.nlevel-3):
                r = self.res(r)
                r_s.append(r)
            for j in reversed(range(1, self.nlevel-3)):
                e_coarse = e_coarse - self.smooth(F.pad(e_coarse, (1, 1, 1, 1), mode='constant', value=0))/self.diag + r_s[j]/self.diag 
                e_coarse = self.prol(e_coarse)
            
            u_update = u - e_coarse - self.smooth(self.boundary_condition(u)) / self.diag + b / self.diag
            u = (1 - self.omega) * u + self.omega * u_update

        return (u-u.min())/(u.max() - u.min())
    def forward(self,t = 4):
        for i in range(1,len(self.u.data)):
            result = self.F_cycle(iteration = t)
            self.u.data[i] = result.detach().numpy()
            self.init_condition = torch.tensor(self.u.data[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.u




class Couple_PDE_Solver(nn.Module):
    def __init__(self,eqs,u,dx,dy,dt,nx,ny,solver,omega=1,s = 0):
        super(Couple_PDE_Solver, self).__init__()
        self.s = s
        A =solver(eqs,u,dt,dx,dy)
        self.target = []
        for eq in eqs:
            for func in u:
                if f'{func.dt}' in str(eq):
                    self.target.append(func)
                    break
        self.target_value = 0
        self.nx = nx
        self.u = u
        self.ny = ny
        self.omega = omega
        self.nlevel = int(math.log(min(nx, ny), 2)) + 1 
        A_res = torch.zeros((1,1,2,2))
        A_res[0,0,:,:] = 0.25
        self.cov = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=0)
        self.res = nn.Conv2d(1,1,kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)
        self.res.weight.data = A_res
        self.diag = 0
        self.A = iter(A)
        self.cov.bias.data = bias_initalizer
        self.res.bias.data = bias_initalizer
        
    def smooth(self,u,weight):
        self.cov.weight.data = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).float()
        return self.cov(u)
    def update_dict(self):
        A_value = next(self.A)
        self.weight_dict = dict(zip(self.u,A_value))
    def generate_diag(self):
        self.target_value = self.target.pop(0)
        i = self.target_value
        self.diag = self.weight_dict[i][1,1]
        self.weight_dict[i][1,1] = 0

    def boundary_condition(self,u):
        
        u_pdd = torch.zeros((1,1, self.ny+2,self.nx+2), dtype=torch.float32)
        u_pdd[0,0,1:-1,1:-1] = u[0,0,:,:]
        u_pdd[0,0,-1,1:-1] = u[0,0,-1,:]
        u_pdd[0,0,0,1:-1] = u[0,0,0,:]
        u_pdd[0,0,1:-1,-1] = u[0,0,:,-1]
        u_pdd[0,0,1:-1,0] = u[0,0,:,0]
        return u_pdd
    
    def F_cycle(self,iteration):
        """
        b is Right-end item (source item) of advection equation
        """
        def couple_eq_sum(weight_dict):
            result = torch.zeros((1,1,self.nx,self.ny),dtype = torch.float32)
            for u, u_w in weight_dict.items():
                u_value = torch.tensor(u.data[0],dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                result += self.smooth(self.boundary_condition(u_value),u_w) 
            return result
        target = self.target_value
        diag = self.diag
        rest_var = self.weight_dict.copy()
        rest_var.pop(target,None)
        b=self.s
        u = torch.tensor(target.data[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        for _ in range(iteration):
            r = couple_eq_sum(self.weight_dict)-b
            r_s = []
            r_s.append(r)
            e_coarse = torch.zeros((1,1,1,1),device = device)
            for i in range(1, self.nlevel-3):
                r = self.res(r)
                r_s.append(r)
            for j in reversed(range(1, self.nlevel-3)):
                e_coarse = e_coarse - (self.smooth(F.pad(e_coarse, (1, 1, 1, 1), mode='constant', value=0), weight = self.weight_dict[target]))/diag + r_s[j]/diag 
                e_coarse = self.prol(e_coarse)
            
            u_update = u - e_coarse - (self.smooth(self.boundary_condition(u), weight = self.weight_dict[target]) + 
                    couple_eq_sum(rest_var)) / diag + b / diag

            u = (1 - self.omega) * u + self.omega * u_update

        return (u-u.min())/(u.max() - u.min())
    def forward(self,t = 4):
        result = []
        for i in range(len(self.u)):
            self.update_dict()
            self.generate_diag()
            result.append(self.F_cycle(t))

        return result
