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
    def __init__(self,eq,u,dx,dy,dt,nx,ny,solver,omega=1):
        super(PDE_Solver, self).__init__()
        A =solver(eq,u,dt,dx,dy)
        A = torch.from_numpy(A).unsqueeze(0).unsqueeze(0).float()
        self.init_condition = torch.tensor(u.data[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.nx = nx
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
            for i in range(1, self.nlevel-5):
                r = self.res(r)
                r_s.append(r)
            for j in reversed(range(1, self.nlevel-5)):
                e_coarse = e_coarse - self.smooth(F.pad(e_coarse, (1, 1, 1, 1), mode='constant', value=0))/self.diag + r_s[j]/self.diag 
                e_coarse = self.prol(e_coarse)
            
            u_update = u - e_coarse - self.smooth(self.boundary_condition(u)) / self.diag + b / self.diag
            u = (1 - self.omega) * u + self.omega * u_update

        return (u-u.min())/(u.max() - u.min())
    def forward(self,t = 4):
        result = self.F_cycle(iteration = t)
        return result


class AI4Diffusion:
    def __init__(self,u,dx,dy,dt,cx,cy,nx,ny,omega = 1):
        super(AI4Diffusion, self).__init__()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.cx = cx
        self.cy = cy
        self.u = u
        self.omega = omega
        self.nx = nx
        self.ny = ny
        self.nlevel = int(math.log(min(nx, ny), 2)) + 1 
        A = torch.zeros((1,1,3,3))
        #A[:,:,0,1] = dt/(2*dy)*cy
        #A[:,:,2,1] = -dt/(2*dy)*cy
        #A[:,:,1,2] = -cx*dt/dx
        A[:,:,1,0] = -cx*dt/dx
        #A[:,:,2,1] = -cy*dt/dy
        A[:,:,0,1] = -cy*dt/dy
        A[:,:,1,1] = 1+cx*dt/dx+cy*dt/dy
        #A[:,:,1,2] = -dt/(2*dx)*cx
        A_res = torch.zeros((1,1,2,2))
        A_res[0,0,:,:] = 0.25
        self.diag = np.array(A)[0,0,1,1]
        A[0,0,1,1] = 0
        self.smooth = nn.Conv2d(1,1,kernel_size=3,stride=1,padding = 0)
        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding = 0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)
        self.smooth.weight.data = A
        self.res.weight.data = A_res
        self.A = A 
    def boundary_condition(self,u):
        u_pdd = torch.zeros((1,1, self.ny+2,self.nx+2))
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
        b=0
        u = self.u.clone()
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

        return u
    def forward(self,t = 4):
        result = self.F_cycle(iteration = t)
        return result


class AI4Advection(nn.Module):
    def __init__(self,u,dx,dy,dt,cx,cy,nx,ny,omega = 1):
        super(AI4Advection, self).__init__()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.cx = cx
        self.cy = cy
        self.u = u
        self.omega = omega
        self.nx = nx
        self.ny = ny
        self.nlevel = int(math.log(min(nx, ny)/128, 2)) + 1 
        A = torch.zeros((1,1,3,3))
        #A[:,:,0,1] = dt/(2*dy)*cy
        #A[:,:,2,1] = -dt/(2*dy)*cy
        #A[:,:,1,2] = -cx*dt/dx
        A[:,:,1,0] = -cx*dt/dx
        #A[:,:,2,1] = -cy*dt/dy
        A[:,:,0,1] = -cy*dt/dy
        A[:,:,1,1] = 1+cx*dt/dx+cy*dt/dy
        #A[:,:,1,2] = -dt/(2*dx)*cx
        A_res = torch.zeros((1,1,2,2))
        A_res[0,0,:,:] = 0.25
        self.diag = np.array(A)[0,0,1,1]
        A[0,0,1,1] = 0
        self.smooth = nn.Conv2d(1,1,kernel_size=3,stride=1,padding = 0)
        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding = 0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)
        self.smooth.weight.data = A
        self.res.weight.data = A_res
        self.A = A 
    def boundary_condition(self,u):
        u_pdd = torch.zeros((1,1, self.ny+2,self.nx+2))
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
        b=0
        u = self.u.clone()
        for _ in range(iteration):
            #r = self.upwind(u) - b
            r = self.smooth(self.boundary_condition(u))-b
            r_s = []
            r_s.append(r)
            e_coarse = torch.zeros((1,1,1,1),device = device)
            for i in range(1, self.nlevel):
                r = self.res(r)
                r_s.append(r)
            for j in reversed(range(1, self.nlevel)):
                e_coarse = e_coarse - self.smooth(F.pad(e_coarse, (1, 1, 1, 1), mode='constant', value=0))/self.diag + r_s[j]/self.diag 
                e_coarse = self.prol(e_coarse)

            u_update = u - e_coarse - self.smooth(self.boundary_condition(u)) / self.diag + b / self.diag
            u = (1 - self.omega) * u + self.omega * u_update

        return u
    def forward(self,t = 4):
        result = self.F_cycle(iteration = t)
        return result


