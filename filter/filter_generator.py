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

