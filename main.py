# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:21:54 2022

@author: wec8371

"""

import torch
from torch.autograd.functional import jacobian
import numpy as np
from timeit import default_timer as timer
from functorch import jacfwd
from functorch import jacrev

def get_capillary(swnew):
    capillary = torch.div(1, swnew)
    capillary[capillary == float("Inf")] = 200
    capillary = 2*swnew
    return capillary


def get_relaperm (swnew):
    
    k0r1=0.6
    L1=1.8
    L2=1.8
    E1=2.1
    E2=2.1
    T1=2.3
    T2=2.3
    
    kr1=torch.div(k0r1*swnew**L1,swnew**L2+E1*(1-swnew)**T1)
    kr2=torch.div((1-swnew)**L2,(1-swnew)**L2+E2*swnew**T2)
    
    return kr1, kr2



def get_residual (unknown):      
    residual = torch.zeros(Np*Nx*Ny*Nz, requires_grad=False, dtype=torch.float64)
    
    
    pre_g = unknown[::2]
    sat_w = unknown[1::2]
    sat_g = 1 - sat_w 
    
    Accumulation_w = sat_w - swold
    Accumulation_g = sat_g - sgold
        
    residual[::2]   += Accumulation_g
    residual[1::2]  += Accumulation_w 
    
    capillary = get_capillary(sat_w)
    pre_w     = pre_g - capillary
    
    gravity_g = rho_g*g
    gravity_w = rho_w*g
    
    
    ##connection list upstream debuging
    #pre_g = torch.rand(Nx*Ny)    
    #checkp = pre_g.detach().numpy()
    #up_glist = torch.zeros(17, requires_grad=True, dtype=torch.float64) 
    
    
    for i in connection_index:
        phi_pre_g = pre_g[connection_a[i]] - pre_g[connection_b[i]]
        phi_pre_w = pre_w[connection_a[i]] - pre_w[connection_b[i]]
        
        up_g = connection_a[i] if phi_pre_g >= 0 else connection_b[i]
        up_w = connection_a[i] if phi_pre_w >= 0 else connection_b[i]

        K_h       = 2*K[connection_a[i]]*K[connection_b[i]] / (K[connection_a[i]] + K[connection_b[i]])
        Tran_h    = K_h*A[i]/d[i]
        Tran_g    = Tran_h*rho_g[up_g]*krg[up_g]/miu_g[up_g]        
        Tran_w    = Tran_h*rho_w[up_w]*krw[up_w]/miu_w[up_w]        
        
        flux_g    = Tran_g*phi_pre_g
        flux_w    = Tran_w*phi_pre_w
        
        ind_a     = 2*connection_a[i]
        ind_b     = 2*connection_b[i]
        

        residual[ind_a] += dt*flux_g
        residual[ind_b] -= dt*flux_g
        
        ind_a     += 1
        ind_b     += 1
        
        #with torch.no_grad():    
        residual[ind_a] += dt*flux_w
        residual[ind_b] -= dt*flux_w
        
    
        
        ##connection list upstream debuging
        #up_glist.data[i] = up_g
        #checkup = up_glist.detach().numpy()
    
    return residual

if __name__ == '__main__':
    
    Nx = 10
    Ny = 10
    Nz = 1
    
    Lx = 10
    Ly = 10
    Lz = 10
    
    
    dx = Lx/Nx
    dy = Ly/Ny
    dz = Lz/Nz
    
     
    dt   = 1
    tf   = 2
    time = 0
    alpha_chop = 0.5
    alpha_grow = 2
    dt_min = 0.1
    Max_iter = 10
    Tol_resi = 1e-7
    
    
    
    g = 9.80665 
    
    Np= 2

    connection_x = torch.arange(0, (Nx-1)*Ny*Nz, requires_grad=False, dtype=torch.int32)
    connection_y = torch.arange(0, Nx*(Ny-1)*Nz, requires_grad=False, dtype=torch.int32)
    connection_z = torch.arange(0, Nx*Ny*(Nz-1), requires_grad=False, dtype=torch.int32) 
    connection   = torch.cat((connection_x, connection_y, connection_z), 0)

    
    A_x  = dy*dz*torch.ones(connection_x.size(dim=0), requires_grad=False, dtype=torch.int32)
    A_y  = dx*dz*torch.ones(connection_y.size(dim=0), requires_grad=False, dtype=torch.int32)
    A_z  = dx*dy*torch.ones(connection_z.size(dim=0), requires_grad=False, dtype=torch.int32)
    A    = torch.cat((A_x, A_y, A_z), 0)
     
   
    d_x  = dx/2*torch.ones(connection_x.size(dim=0), requires_grad=False, dtype=torch.int32)
    d_y  = dy/2*torch.ones(connection_y.size(dim=0), requires_grad=False, dtype=torch.int32)
    d_z  = dz/2*torch.ones(connection_z.size(dim=0), requires_grad=False, dtype=torch.int32)
    d    = torch.cat((d_x, d_y, d_z), 0)
    

    connection_x_index = torch.arange(0, (Nx-1)*Ny*Nz, requires_grad=False, dtype=torch.int32)
    connection_y_index = torch.arange((Nx-1)*Ny*Nz, (Nx-1)*Ny*Nz+Nx*(Ny-1)*Nz, requires_grad=False, dtype=torch.int32)
    connection_z_index = torch.arange((Nx-1)*Ny*Nz+Nx*(Ny-1)*Nz, (Nx-1)*Ny*Nz+Nx*(Ny-1)*Nz+Nx*Ny*(Nz-1), requires_grad=False, dtype=torch.int32)
    connection_index  = torch.cat((connection_x_index, connection_y_index, connection_z_index), 0)
    #check=connection_index.detach().numpy()  

   
    connection_xa = connection_x + torch.div(connection_x, Nx-1, rounding_mode='trunc')
    connection_xb = connection_xa + 1
    connection_ya = connection_y + Nx*torch.div(connection_y, Nx*(Ny-1), rounding_mode='trunc')
    connection_yb = connection_ya + Nx
    connection_za = connection_z
    connection_zb = connection_za + Nx*Ny
    #check2=connection_za.detach().numpy()
    #check3=connection_zb.detach().numpy()
    
    connection_a  = torch.cat((connection_xa, connection_ya, connection_za), 0)
    connection_b  = torch.cat((connection_xb, connection_yb, connection_zb), 0)
   
    
    # Debug
    swnew     = torch.ones(Nx*Ny*Nz, requires_grad=True, dtype=torch.float64)   
    swold     = torch.zeros(Nx*Ny*Nz, requires_grad=True, dtype=torch.float64)    
    sgold     = 1 - swold    
    
    pgnew     = torch.zeros(Nx*Ny*Nz, requires_grad=True, dtype=torch.float64)
    pgold     = torch.zeros(Nx*Ny*Nz, requires_grad=True, dtype=torch.float64)
    pc        = get_capillary(swnew)
    unknown   = torch.ravel(torch.column_stack((pgnew, swnew)))
    

    
    krw       = get_relaperm (swnew)[0]
    krg       = get_relaperm (swnew)[1]
    K         = torch.ones(Nx*Ny*Nz, requires_grad=True, dtype=torch.float64)
    miu_w     = torch.ones(Nx*Ny*Nz, requires_grad=True, dtype=torch.float64)
    miu_g     = torch.ones(Nx*Ny*Nz, requires_grad=True, dtype=torch.float64)
    rho_w     = 1000*torch.ones(Nx*Ny*Nz, requires_grad=True, dtype=torch.float64)
    rho_g     = 2*torch.ones(Nx*Ny*Nz, requires_grad=True, dtype=torch.float64)
    
    
    '''
    start = timer()
    r      = get_residual(unknown)
    end = timer()
    print('get_residual ', end - start)
    
    
    start = timer()
    J_funcfwd    = jacfwd(get_residual)(unknown)
    end = timer()
    print('jacobian(jacfwd) ', end - start)
    '''

    

    
    
    while abs(time - tf) > 1e-8:
        niter  = 0
        r      = get_residual(unknown)
        J      = jacfwd(get_residual)(unknown)
        
        while True:
            update       = torch.linalg.solve(J, r)
            unknown     -= update
            r            = get_residual(unknown)
            niter       += 1
            
            if (torch.linalg.vector_norm(r) <= Tol_resi):
                is_coverged = 1
            else:
                is_coverged = 0
                J      = jacfwd(get_residual)(unknown)
                
            
            if ((niter > Max_iter) or (is_coverged)):
                break

        if (not is_coverged):
            dt    *= alpha_chop 
            dt     = dt if dt >= dt_min else dt_min
        else:
            time  += dt
            dt    *= alpha_grow
            dt     = (tf - time) if (time + dt) > tf else dt

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    