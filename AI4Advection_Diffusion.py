import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")


class AI4Advection(nn.Module):
    def __init__(self,u,dx,dy,dt,cx,cy,nx,ny):
        super(AI4Advection, self).__init__()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.cx = cx
        self.cy = cy
        self.u = u
        self.nx = nx
        self.ny = ny
        self.nlevel = int(math.log(min(nx, ny), 2)) + 1 
        A = torch.zeros((1,1,3,3))
        A[:,:,0,1] = self.dt/self.dy/2*self.cy
        A[:,:,2,1] = -self.dt/self.dy/2*self.cy
        A[:,:,1,0] = self.dt/self.dx/2*self.cx
        A[:,:,1,1] = 1
        A[:,:,1,2] = -self.dt/self.dx/2*self.cx
        A_res = torch.zeros((1,1,2,2))
        A_res[0,0,:,:] = 0.25
        self.diag = np.array(A)[0,0,1,1]
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

            u = u - e_coarse
            u = u - self.smooth(self.boundary_condition(u)) / self.diag + b / self.diag
        return u
    def forward(self,t = 4):
        result = self.F_cycle(iteration = t)
        return result


class AI4Diffusion(nn.Module):
    def __init__(self,u,dx,dy,dt,D,nx,ny):
        super(AI4Diffusion, self).__init__()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.u = u
        self.nx = nx
        self.ny = ny
        self.nlevel = int(math.log(min(nx, ny), 2)) + 1 
        A = torch.zeros((1,1,3,3))
        A[:,:,0,1] = self.dt/self.dy**2*D
        A[:,:,2,1] = self.dt/self.dy**2*D
        A[:,:,1,0] = self.dt/self.dx**2*D
        A[:,:,1,1] = 1-2*(self.dt/self.dy**2*D + self.dt/self.dx**2*D)
        A[:,:,1,2] = self.dt/self.dx**2*D
        A_res = torch.zeros((1,1,2,2))
        A_res[0,0,:,:] = 0.25
        self.diag = np.array(A)[0,0,1,1]
        self.smooth = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=0)
        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)
        self.smooth.weight.data = A
        self.res.weight.data = A_res
        
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
            u = u - e_coarse
            u = u - self.smooth(self.boundary_condition(u)) / self.diag + b / self.diag
        return u
    def forward(self,t = 4):
        result = self.F_cycle(iteration = t)
        return result




