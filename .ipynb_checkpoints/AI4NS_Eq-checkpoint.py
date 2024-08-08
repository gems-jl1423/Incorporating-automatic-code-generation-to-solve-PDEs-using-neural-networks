import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sympy as sp
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")


def coupling_eq(eqs,functions,delta_t,dx,dy):
    couples = []
    matrixs = [[] for i in range(3)]
    matrix_A = []
    z = sp.symbols('z')
    dt = sp.symbols('dt')
    def discretization(eq,u):
        t,x,y  = u.dimensions
        eq = eq.subs({
            u.dx: (-u.forward.shift(x, -x.spacing) + u.forward)/dx,  
            u.dy: (-u.forward.shift(y, -y.spacing) + u.forward)/dy,
            u.dx2: (u.forward.shift(x,x.spacing) - 2*u.forward + u.forward.shift(x, -x.spacing))/dx**2,
            u.dy2: (u.forward.shift(y,y.spacing) - 2*u.forward + u.forward.shift(y, -y.spacing))/dy**2
            })
        return eq

    for eq in eqs:
        pde = eq.copy()
        target = None
        for u in functions:
            pde = discretization(pde,u)
            if f'{u.dt}' in str(pde):
                target = u
        pde = pde.subs(target.dt, z)
        for u in functions:
            pde = pde.subs(u,u.forward)
        pde=pde.subs(z,target.dt)
        pde = pde.subs(target.dt,(target.forward - target)/delta_t)
        pde = solve(pde,target)
        stencil_fd = pde.simplify()
        coeff_dict = stencil_fd.as_coefficients_dict()
        str_coeff_dict = {str(key): value for key, value in coeff_dict.items()}
        couples.append(str_coeff_dict)
    
    for eq in couples:
        varibles = []
        for u in functions:
            t,x,y = u.dimensions
            matrix = np.zeros((3,3))
            keys = np.empty((3,3), dtype='<U100')
            keys[0,0] = f'{u.forward.shift(x,-x.spacing).shift(y,+y.spacing)}'
            keys[0,1] = f'{u.forward.shift(y,+y.spacing)}'
            keys[0,2] = f'{u.forward.shift(x,+x.spacing).shift(y,+y.spacing)}'
            keys[1,0] = f'{u.forward.shift(x,-x.spacing)}'
            keys[1,1] = f'{u.forward}'
            keys[1,2] = f'{u.forward.shift(x,+x.spacing)}'
            keys[2,0] = f'{u.forward.shift(x,-x.spacing).shift(y,-y.spacing)}'
            keys[2,1] = f'{u.forward.shift(y,-y.spacing)}'
            keys[2,2] = f'{u.forward.shift(x,+x.spacing).shift(y,-y.spacing)}'
            for i in range(len(keys)):
                for j in range(len(keys[i])):
                    key = keys[i][j]
                    if key in eq:
                        matrix[i,j] = eq[key] 
            varibles.append(matrix)
        matrix_A.append(varibles)
    for i in range(3):
        for j in range(3):
            small_martix = np.zeros((len(functions),len(functions)))
            for k in range(len(couples)):
                for y in range(len(functions)):
                    small_martix[k][y] = matrix_A[k][y][i,j]
            matrixs[i].append(small_martix)

    return np.array(matrix_A)


bias_initalizer = taorch.tensor([0], dtype=torch.float32)


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
