from devito import *
import sympy as sp
import numpy as np
def centre_difference(eq,u,delta_t,dx,dy):
    stencil = solve(eq, u)
    #stencil = Eq(u.forward, stencil)
    #stencil = solve(stencil, u)
    t,x,y = u.dimensions
    dt = sp.symbols('dt')
    stencil_fd = stencil.subs({
    dt:delta_t,
    u.dx: (u.forward.shift(x, x.spacing) - u.forward.shift(x, -x.spacing))/2/dx,  # x方向偏移1
    u.dy: (u.forward.shift(y, y.spacing) - u.forward.shift(y, -y.spacing))/2/dy,  # y方向偏移1
    u.dx2: (u.forward.shift(x,x.spacing) - 2*u.forward + u.forward.shift(x, -x.spacing))/dx**2,
    u.dy2: (u.forward.shift(y,y.spacing) - 2*u.forward + u.forward.shift(y, -y.spacing))/dy**2
    })
    stencil_fd = stencil_fd.simplify()
    coeff_dict = stencil_fd.as_coefficients_dict()
    str_coeff_dict = {str(key): value for key, value in coeff_dict.items()}
    matrix = np.zeros((3,3))
    keys = np.empty((3,3), dtype='<U100')
    keys[0,0] = f'u({t} + {delta_t}, {x} - {x.spacing}, {y} - {y.spacing})'
    keys[0,1] = f'u({t} + {delta_t}, {x}, {y} - {y.spacing})'
    keys[0,2] = f'u({t} + {delta_t}, {x} + {x.spacing}, {y} - {y.spacing})'
    keys[1,0] = f'u({t} + {delta_t}, {x} - {x.spacing}, {y})'
    keys[1,1] = f'u({t} + {delta_t}, {x}, {y})'
    keys[1,2] = f'u({t} + {delta_t}, {x} + {x.spacing}, {y})'
    keys[2,0] = f'u({t} + {delta_t}, {x} - {x.spacing}, {y} + {y.spacing})'
    keys[2,1] = f'u({t} + {delta_t}, {x}, {y} + {y.spacing})'
    keys[2,2] = f'u({t} + {delta_t}, {x} + {x.spacing}, {y} + {y.spacing})'
    for i in range(len(keys)):
        for j in range(len(keys[i])):
            key = keys[i][j]
            if key in str_coeff_dict:
                matrix[i,j] = str_coeff_dict[key] 
    return matrix
def upwind(eq,u,delta_t,dx,dy):
    stencil = solve(eq, u)
    t,x,y = u.dimensions
    dt = sp.symbols('dt')
    stencil_fd = stencil.subs({
    dt:delta_t,
    u.dx: (-u.forward.shift(x, -x.spacing) + u.forward)/dx,  
    u.dy: (-u.forward.shift(y, -y.spacing) + u.forward)/dy})
    stencil_fd = stencil_fd.simplify()
    coeff_dict = stencil_fd.as_coefficients_dict()
    str_coeff_dict = {str(key): value for key, value in coeff_dict.items()}
    matrix = np.zeros((3,3))
    keys = np.empty((3,3), dtype='<U100')
    keys[0,0] = f'u({t} + {delta_t}, {x} - {x.spacing}, {y} - {y.spacing})'
    keys[0,1] = f'u({t} + {delta_t}, {x}, {y} - {y.spacing})'
    keys[0,2] = f'u({t} + {delta_t}, {x} + {x.spacing}, {y} - {y.spacing})'
    keys[1,0] = f'u({t} + {delta_t}, {x} - {x.spacing}, {y})'
    keys[1,1] = f'u({t} + {delta_t}, {x}, {y})'
    keys[1,2] = f'u({t} + {delta_t}, {x} + {x.spacing}, {y})'
    keys[2,0] = f'u({t} + {delta_t}, {x} - {x.spacing}, {y} + {y.spacing})'
    keys[2,1] = f'u({t} + {delta_t}, {x}, {y} + {y.spacing})'
    keys[2,2] = f'u({t} + {delta_t}, {x} + {x.spacing}, {y} + {y.spacing})'
    for i in range(len(keys)):
        for j in range(len(keys[i])):
            key = keys[i][j]
            if key in str_coeff_dict:
                matrix[i,j] = str_coeff_dict[key] 
    return matrix
def DA_eq(eq,u,delta_t,dx,dy):
    stencil = solve(eq, u)
    #stencil = Eq(u.forward, stencil)
    #stencil = solve(stencil, u)
    t,x,y = u.dimensions
    dt = sp.symbols('dt')
    stencil_fd = stencil.subs({
    dt:delta_t,
    u.dx: (-u.forward.shift(x, -x.spacing) + u.forward)/dx,  
    u.dy: (-u.forward.shift(y, -y.spacing) + u.forward)/dy,
    u.dx2: (u.forward.shift(x,x.spacing) - 2*u.forward + u.forward.shift(x, -x.spacing))/dx**2,
    u.dy2: (u.forward.shift(y,y.spacing) - 2*u.forward + u.forward.shift(y, -y.spacing))/dy**2
    })
    stencil_fd = stencil_fd.simplify()
    coeff_dict = stencil_fd.as_coefficients_dict()
    str_coeff_dict = {str(key): value for key, value in coeff_dict.items()}
    matrix = np.zeros((3,3))
    keys = np.empty((3,3), dtype='<U100')
    keys[0,0] = f'u({t} + {delta_t}, {x} - {x.spacing}, {y} + {y.spacing})'
    keys[0,1] = f'u({t} + {delta_t}, {x}, {y} + {y.spacing})'
    keys[0,2] = f'u({t} + {delta_t}, {x} + {x.spacing}, {y} + {y.spacing})'
    keys[1,0] = f'u({t} + {delta_t}, {x} - {x.spacing}, {y})'
    keys[1,1] = f'u({t} + {delta_t}, {x}, {y})'
    keys[1,2] = f'u({t} + {delta_t}, {x} + {x.spacing}, {y})'
    keys[2,0] = f'u({t} + {delta_t}, {x} - {x.spacing}, {y} - {y.spacing})'
    keys[2,1] = f'u({t} + {delta_t}, {x}, {y} - {y.spacing})'
    keys[2,2] = f'u({t} + {delta_t}, {x} + {x.spacing}, {y} - {y.spacing})'
    for i in range(len(keys)):
        for j in range(len(keys[i])):
            key = keys[i][j]
            if key in str_coeff_dict:
                matrix[i,j] = str_coeff_dict[key] 
    return matrix

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
