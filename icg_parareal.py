#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:48:20 2023

"""

import numpy as np
import scipy.linalg as splin
import math
import matplotlib.pyplot as plt

import plots as pf

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                        Parareal                                             #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Parareal(F,G,lambda0,N,k=None,eps=None,guess=None):
    """
    Parareal algorithm
    arguments:
        F     - fine solver
        G     - coarse solver
        N     - no. of time windows
        k     - parareal iterations
        eps   - stopping tolerance
        guess - initial guess 
    """

    if guess is not None:
        maxind = 1
    else:
        maxind = N
    lambdank = np.zeros((N+1,maxind+1,lambda0.shape[0]),dtype = type(lambda0))
    Flambdank = np.zeros(((N+1)*maxind+1,lambda0.shape[0]),dtype = type(lambda0))
    lambdank[0,0] = lambda0 

# Intitialisation by coarse solver
    if guess is not None:
        lambdank[:,0] = guess[:,k]    
        maxind = 1
    else:
        if k != None:
            maxind = k
        p=0
        for n in range(1,N+1):
            lambdank[n,p] = np.dot(G,lambdank[n-1,p])

# k parareal iterations
    for p in range(1,maxind+1):
        # print('Iteration: {}'.format(p))
        lambdank[0,p] = lambda0
        for n in range(1,N+1):
            lambdank[n,p] = np.dot(F,lambdank[n-1,p-1]) - np.dot(G,lambdank[n-1,p-1]) + np.dot(G,lambdank[n-1,p])
        if eps != None:
            err = np.linalg.norm(lambdank[N,p]-lambdank[N,p-1])
            print('Error: {}'.format(err))
            if err <= eps:
                break
        
    return lambdank,p

def Finesolution(F,lambda0,N):
    """
    Reference solution
    obtained by running the fine solver F for initial state lambda0
    """

    lambdank = np.zeros((N+1,lambda0.shape[0]),dtype = type(lambda0))
    lambdank[0] = lambda0
    for n in range(1,N+1):
        lambdank[n] = np.dot(F,lambdank[n-1])
        
    return lambdank


def Pararealerror(F,G,N,k):
    res = np.zeros_like(F)
    for p in range(k+1,N+1):
        res = res + math.comb(N,p)*np.matmul(np.linalg.matrix_power(F-G,p),np.linalg.matrix_power(G,N-p))
        
    return res


# Matrices associated with data assimilation problem #
def Amatrix(F,G,N,x,k=None,eps=None):
    """
    Matrix of associated system Amatrix = (F^N)^T * F^N
    returns Amatrix*x
    """

    if multi_obs:
        res = np.zeros(C.shape[0])
        for i in range(len(multi_N)):
            res = res + np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),np.linalg.matrix_power(F,multi_N[i]))
        res = res/(len(multi_N))
    else:        
        res = np.matmul(np.transpose(np.linalg.matrix_power(F,N)),np.linalg.matrix_power(F,N))
    if regularisation:
        if multi_obs:
            res = res + alpharegul*np.identity(res.shape[0])
        else:
            res = res + alpharegul*np.identity(res.shape[0])
    return np.dot(res,x)


def APmatrix(F,G,N,x,k=None,eps=None):
    """
    A matrix of the associated linear system when one uses Parareal for the forward model
    and the true model F for the backward model.
    APmatrix = (F^N)^T * Parareal
    returns APmatrix*x
    """

    par,it = Parareal(F,G,x,N,k,eps)
    print('Iterations: ',it)
    
    if multi_obs:
        res = np.zeros(C.shape[0])
        for i in range(len(multi_N)):
            res = res + np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),par[multi_N[i],it])
        res = res/(len(multi_N))
    else:
        res = np.matmul(np.transpose(np.linalg.matrix_power(F,N)),par[N,it])
    if regularisation:
        if multi_obs:
            res = res +alpharegul*x 
            #res = res + alpharegul*np.matmul(pen_matrix(),x)
        else:
            res = res + alpharegul*x
    
    return res


def Ematrix(F,G,N,k):
    """
    Error matrix: difference between APmatrix and Amatrix
    E(k) = (F^N)^T(F^N -P(k))
    """

    if multi_obs:
        res = np.zeros(C.shape[0])
        for i in range(len(multi_N)):
            res = res - np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),Pararealerror(F,G,multi_N[i],k))
    else:
        res = -np.matmul(np.transpose(np.linalg.matrix_power(F,N)),Pararealerror(F,G,N,k))          

    return res



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                              Conjugate gradient                                             #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def conjgrad(A,F,G,N,b,x,k=None,epspara=None,eps=1.e-6):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix 
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """ 
    
    # arrays to store values
    res = []                                        # 2-norm of residual
    para_it = []                                    # number of parareal iterations
    
    r = b -A(F,G,N,x,k,epspara)
    beta_old = np.square(np.linalg.norm(b,2))
    u=np.zeros((2*N,x.shape[0]))
    u[0,:] = b/beta_old
    p = r
    rsold = np.dot(np.transpose(r),r)

    reorth = True                                   # reorthogonalization or not
    iteration = 0
    while True:
        print("########## CG iteration {} ###########".format(iteration))

        if (A == Amatrix):
            Ap = A(F,G,N,p,k,epspara)
        
        if (A == APmatrix):
            ap,it = Parareal(F,G,p,N,eps=1.e-6)
            ap1 = ap[N,it]
            Ap = np.matmul(np.transpose(np.linalg.matrix_power(F,N)),ap1)
            para_it.append(it)
        
        alpha = rsold/np.dot(np.transpose(p),Ap)
        x = x + np.dot(alpha,p)
        r = r -np.dot(alpha,Ap)
        rsnew = np.dot(np.transpose(r),r)
        res.append(np.sqrt(rsnew))
        print('2-norm of r: ', res[iteration])
        
        if np.sqrt(rsnew) <eps:
            break
        if reorth:
         	for i in range(iteration+1):
                  r = r -np.dot(u[i,:],r)*u[i,:]
         	beta_new = np.matmul(np.transpose(r),r)
         	u[iteration+1,:]=r/np.sqrt(beta_new)
        else:
            beta_new = np.matmul(np.transpose(r),r)
        p = r + (beta_new/beta_old)*p
        rsold = rsnew
        beta_old = beta_new
        iteration+=1

    return x, iteration+1, res, para_it



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                         Data assimilation and minimisation                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#------- regularisation based on spatial derivative of the initial state -------#
regularisation = True                                                           #
alpharegul = 1.e-5                                                              #
#-------------------------------------------------------------------------------#

# matrix obtained from the regularisation term
def regul_matrix(C):
    mat = np.zeros_like(C)
    mat[len(mat)-1,len(mat)-1] = 2
    for i in range(len(lambda0)-1):
        mat[i,i] = 2
        mat[i,i+1] = -1
        mat[i+1,i] = -1
    return -mat*(1/dx**2)

# Gradient/Residual for Ax = b
def residual(x):
    grad = np.matmul(A,x) - b
    return grad

# ellipsoidal norm
def ep_norm(P,x):
    return np.sqrt(np.matmul(np.transpose(x),np.matmul(P,x)))

# quadratic form, q(x) = 0.5*(x^T A x) - b^T x
def quad(x):
    q1 = 0.5*(np.matmul(np.transpose(x),np.matmul(A,x))) - np.matmul(np.transpose(b),x)
    return q1


# estimates from Gratton et al.

# bound for E_norm
def E_bound(eps,b,p,r,phi):
    eps1 = np.sqrt(eps)
    b1 = ep_norm(A_inv,b)
    p1 = ep_norm(A,p)  
    r1 = np.square(np.linalg.norm(r,2))
    
    omega = (eps1*b1*p1)/(2*phi*r1 + eps1*b1*p1)
    return omega

# Primal dual matrix norm for perturbation
def E_norm(B,k):
    P = splin.sqrtm(B)
    P_inv = np.linalg.inv(P)
    E = np.matmul(np.matmul(P_inv,Ematrix(F,G,N,k)),P_inv)   
   
    return np.linalg.norm(E,2)     

def prac_Enorm(E):
    lam = 1/min(np.abs(np.linalg.eig(A)[0]))
    return lam*np.linalg.norm(E,2)

def prac_pnorm(p):
    return np.sqrt(np.trace(A)/A.shape[0])*np.linalg.norm(p,2)

def inexact_quad(x):
    return -0.5*np.matmul(np.transpose(b),x)

def prac_bnorm(x,it):
    if it == 0:
        return np.linalg.norm(b,2)/np.sqrt(max(np.abs(np.linalg.eig(A)[0])))
    else:
        return np.sqrt(2*abs(inexact_quad(x)))
    
def prac_Ebound(eps,x,p,r,phi,it):
    
    eps1 = np.sqrt(eps)
    if it == 0:
        q1 = (np.sqrt(2)*np.linalg.norm(b,2))/(np.sqrt(max(np.abs(np.linalg.eig(A)[0]))))
    else:
        q1 = np.sqrt(np.abs(inexact_quad(x)))
    t1 = np.sqrt(np.trace(A))
    p1 = np.linalg.norm(p,2)
    r1 = np.square(np.linalg.norm(r,2))
    
    return (eps1*q1*t1*p1)/(np.sqrt(2*A.shape[0])*phi*r1 + eps1*q1*t1*p1)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                 Inexact conjugate gradient                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def inexact_CG(b,eps,x,nitermax):
    
    qd = []                                         # quadratic difference
    gap1 = []                                       # residual gap
    res = [-b]                                      # inexact residual
    res_norm = []                                   # inexact residual norm
    e_bd = []                                       # omega
    para_iter = []                                  # parareal iteration array
    pvec = [b]                                      # direction vector
    e_bd2 = []                                      # omega2    
    
    beta_old = np.square(np.linalg.norm(b,2))
    r = -b
    p = b
    
    #-------- integer for the practical stopping criterion ---------#
    d = 2                                                           #
    #---------------------------------------------------------------#
    
    q = [0]                                         # quadratic values
    bound = 0.5*np.sqrt(eps)*ep_norm(A_inv, b)      # theoretical tolerance
    print('Minimisation tolerance ',bound )
    iteration = 0    
    phi = nitermax
    big_phi = 1
    print('nitermax = ',nitermax)

    reorth = True                                   # Reorthogonalization or not
    u=np.zeros((nitermax,x.shape[0]))
    u[0,:] = b/beta_old
        
    #----------------------------------------------------------------------------------#
    # options (choose one):                                                            #
    # 1. use original parareal criterion, || E ||_A^{-1},A                             #
    # 2. use modified parareal criterion, || Ep ||_A^{-1}                              #
    # 3. use all practical approximations including the estimate for || Ep ||_A^{-1}   #
    #----------------------------------------------------------------------------------#
    use_enorm = False
    use_epnorm = False
    use_all_approx = True


    while True:
        print('#########################################')
        print('Iteration {} '.format(iteration),'\n')
        print('phi : ', phi)
        
        if (use_enorm or use_epnorm):
            omega = E_bound(eps,b,p,r,phi)
            omega2 = omega*ep_norm(A,p)
            inacc_budget = False

        if (use_all_approx):
            omega = prac_Ebound(eps,x,p,r,phi,iteration)
            omega2 = omega*prac_pnorm(p)
            inacc_budget = True

        print("omega: ", omega)
        e_bd.append(omega)
        print("omega x ||p||: ",omega2)
        e_bd2.append(omega2)

        #with || E ||_A^{-1},A 
        if (use_enorm):
            k2 = 1
            while k2<N:
                e = E_norm(A,k2)
                print('Error at parareal iteration ',k2,' : ',e)
                if (e>omega and k2!=N-1):
                    k2 = k2+1
                   
                elif k2==N-1:
                    print('Number of parareal iterations : ',k2)
                    omega_hat = E_norm(A,k2)
                    print('omega_hat : ',omega_hat)
                    para_iter.append(k2)
                    c = APmatrix(F,G,N,p,k2)
                    break
                else:
                    print('Number of parareal iterations : ',k2)
                    omega_hat = E_norm(A,k2)
                    print('omega_hat : ',omega_hat)
                    para_iter.append(k2)
                    c = APmatrix(F,G,N,p,k2)
                    break

        
        # with exact || Ep ||_A^{-1}
        if (use_epnorm):
            k1=1
            while k1<=N:
                # e = ep_norm(A_inv,APmatrix(F,G,N,p,k1) - np.dot(A,p))
                # e = ep_norm(A_inv,np.dot(Ematrix(F,G,N,k1),p))
                e = ep_norm(A_inv,APmatrix(F,G,N,p,k1)-Amatrix(F,G,N,p))
                print('|| Ep || at parareal iteration ',k1,' : ',e)
                if e > omega2:
                    k1 = k1+1
                else:
                    print('Number of parareal iterations : ',k1)
                    omega_hat = e
                    print('omega_hat: ',omega_hat)
                    para_iter.append(k1)
                    c = APmatrix(F,G,N,p,k1)
                    break
               


        # using approximate || Ep ||_A^{-1} and all other approximations (practical)
        if (use_all_approx):
            x_est, ksol = Parareal(F,G,p,N,k=10)
            p_star = np.matmul(np.linalg.matrix_power(F,N),p)
            p_vals = []
            ep_vals = []
            for i in range(1,N):
                # x_est, ksol = Parareal(F,G,p,N,k=2)
                e = np.linalg.norm(x_est[N,i+1] - x_est[N,i],2)
                e1 = np.linalg.norm(x_est[N,i] - p_star)
                p_vals.append(e1)
                ep_vals.append(e)
                print('Ep norm after', i+1, ' parareal iterations : ', e)
                if (e < omega2):
                    omega_hat = e
                    print('omega hat : ', omega_hat)
                    print('parareal iterations ', i+1)
                    ksol=i+1
                    para_iter.append(ksol)
                    break

            #print('ksol for c calculation', ksol)
            if multi_obs:
                c = np.zeros(A.shape[0])
                for i in range(len(multi_N)):
                    c = c + np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),x_est[multi_N[i],ksol])
            else:
                c = np.matmul(np.transpose(np.linalg.matrix_power(F,N)),x_est[N,ksol])


        if regularisation:
            if multi_obs:
                c = c + alpharegul*np.matmul(pen_matrix(),p)
            else:
                c = c + alpharegul*p


        if inacc_budget:
            phi_hat = ((prac_pnorm(p)-omega_hat)*np.sqrt(eps)*prac_bnorm(x,iteration)*prac_pnorm(p))/(omega_hat*2*np.square(np.linalg.norm(r,2)))
            print('phi : ',phi,'\nphi_hat : ',phi_hat,'\nphi_hat > phi : ',phi_hat>phi)
            big_phi_new = big_phi - (1/phi_hat)
            if iteration<nitermax:
                phi_new = (nitermax-iteration-1)/big_phi_new
            else:
                phi_new = phi
            phi = phi_new
            big_phi = big_phi_new

        res_gap = ep_norm(A_inv,residual(x)-r)
        gap1.append(res_gap)
        # bound for quadratic
        quad_bound = eps*np.abs(quad(x_star))
        alpha = beta_old/np.matmul(np.transpose(p),c)
        x_new = x + np.dot(alpha,p)
        r_new = r + np.dot(alpha,c)

        q.append(inexact_quad(x_new))
        quad_diff = np.abs(quad(x_new)-quad(x_star))
        qd.append(quad_diff)
        print('2-norm of r: ', np.linalg.norm(r_new))
        print('A-1 norm of r: ', ep_norm(A_inv,r_new), '\n')
        
        res_norm.append(ep_norm(A_inv,r_new))
        
        
        #------------------------------- inexact CG stopping criterion ---------------------------------#
        # stopping criterion using ||r_j||_A^{-1}                                                       #
        if (use_enorm or use_epnorm):                                                                   #
            if ep_norm(A_inv,r_new) <= bound:                                                           #
                break                                                                                   #
        # using approximations                                                                          #
        if (use_all_approx):                                                                            #
            if iteration >=d:                                                                           #
                if (q[iteration +1 -d] - q[iteration+1]) <= 0.25*eps*np.abs(inexact_quad(x_new)):       #
                     break                                                                              #
        #-----------------------------------------------------------------------------------------------#

        # reorthogonalisation
        if reorth:
        	for i in range(iteration+1):
        		r_new = r_new -np.dot(u[i,:],r_new)*u[i,:]
        	beta_new = np.matmul(np.transpose(r_new),r_new)
        	u[iteration+1,:]=r_new/np.sqrt(beta_new)
        else:
            beta_new = np.matmul(np.transpose(r_new),r_new)
            
        p_new = -r_new + np.dot((beta_new/beta_old),p)
        
        # update old values
        beta_old = beta_new
        r = r_new
        x = x_new
        p = p_new
        
        pvec.append(p)
        iteration+=1
            
    return x_new, iteration+1,res_norm, qd, bound, quad_bound, gap1, e_bd, para_iter,pvec,e_bd2

    
#~~~~~~~~~~~~~~~~~~~~~~ Analysis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#----------------------- Parareal parameters -----------------------------#   
omegamax = 0.29997 # Courant number                                       #
N = 20  # Set N number of time windows                                    #
                                                                          #
# Compute a dt such that max(|eigenvalues(A)|)*dt = omegamax              #
# dt = omegamax/np.max(np.abs(l0))                                        #
# Dt = dt * Nfine / q                                                     #
                                                                          #
# q has to divide Nfine                                                   #
dt = 0.05           # fine time-step                                      #
Nfine = 100         # no. of fine time steps per time windows             #
q=20                # no. of coarse time steps per time windows           #
Dt = dt*Nfine/q     # coarse time-step                                    #
theta = 0.51                                                              #
#-------------------------------------------------------------------------#

#------------------- Shallow water model parameters ----------------------#
H=0.9               # mean height                                         #
g=10.               # gravity                                             #
L=120.              # horizontal length                                   #
nx = 120            # no. of spatial  grid points                         #
dx=L/nx             # spatial resolution                                  #
visc = 0.15         # viscosity                                           #
#-------------------------------------------------------------------------# 


# Shallow Water model
# X(0:nx-1) = eta(1:nx) (free surface)
# X(nx:2*nx-2) = u(1:nx-1)

# staggered grid 
# for i=0 to nx-1
#    u(i-1)          | eta(i)    | u(i)
#    X(nx+i-1)       |  X(i)     | X(nx+i)

# staggered grid
# for i=0 to nx-2
#   eta(i)    | u(i)      | eta(i+1)
#    X(i)     | X(nx+i)   |  X(i+1)


print("\n ------Diffusion courant number------", (visc*dt)/(dx**2),'\n')
C = np.zeros((2*nx-1,2*nx-1))

# d eta/dt = -H * du/dx
for i in range(nx):
    if (i==0):
        C[i,nx+i]=-H/dx
    elif (i==nx-1):
        C[i,nx+i-1]=H/dx
    else:
        C[i,nx+i]=-H/dx
        C[i,nx+i-1]=H/dx

for i in range(nx-1):
    C[nx+i,i+1]=-g/dx
    C[nx+i,i]=g/dx
    if (i!=0):
        C[nx+i,nx+i-1] = visc/dx**2
    if (i!=nx-2):
        C[nx+i,nx+i+1] = visc/dx**2
    C[nx+i,nx+i] = (-2*visc)/dx**2
    

# xr(0:nx-1) : locations of eta points
xr=np.zeros(nx)
for i in range(nx):
    xr[i]=i*dx+dx/2.

l0,v = np.linalg.eig(C)
l1 = np.abs(l0)
print('Conditioning of C matrix = ',np.max(l1)/np.min(l1))

F = np.linalg.matrix_power(np.matmul(np.linalg.inv(np.identity(C.shape[0])-theta*(dt)*C),np.identity(C.shape[0])+(1.-theta)*(dt)*C),Nfine)
G = np.linalg.matrix_power(np.matmul(np.linalg.inv(np.identity(C.shape[0])-theta*(Dt)*C),np.identity(C.shape[0])+(1.-theta)*(Dt)*C),q)

# propagator norms
print('||F||_2 = {}, ||G||_2 = {}, \n||F-G||_2 = {}'.format(np.linalg.norm(F,2),np.linalg.norm(G,2),np.linalg.norm(F-G,2)),'\n')
print('Total integration time = ',dt*Nfine*N)

lambda0 = np.zeros(C.shape[0],dtype=C.dtype)
# Initialize a Gaussian at the middle of the basin
lambda0[:]=0.
for i in range(nx):
    lambda0[i]=np.exp(-((xr[i]-L/2.)/(L/15.))**2) #*np.sin(20.*xr[i]/L)



# reference solution
refsol = Finesolution(F,lambda0,N)[N]

# parareal solution
lambdank,it = Parareal(F,G,lambda0,N,eps=1.e-6)
print('Iterations',it)

# solution plot
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (12,6))
# free surface
ax1.plot(xr,lambdank[N,it][0:nx])
ax1.set_title('free surface')
# velocity
ax2.plot(lambdank[N,it][nx:])
ax2.set_title('velocity')
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~ Analysis for inexact CG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#-------------------- use single/multiple observations -------------------------#
multi_obs = False                                                               #
#time windows where observations are defined                                    #
# multi_N = [10*i for i in range(1,3)]                                          #
multi_N = [4*i for i in range(1,6)]                                             #
                                                                                #
N_obs = len(multi_N)                                                            #
                                                                                #
# True observation at the end of integration time                               #
#if multi_obs:                                                                  #
#    obs = [Finesolution(F,lambda0,n)[n] for n in multi_N]                      #    
#else:                                                                          #    
#    obs = Finesolution(F,lambda0,N)[N]                                         #    
                                                                                #        
# OR using the exact solution                                                   #    
if multi_obs:                                                                   #
    obs = [np.matmul(splin.expm(C*(n*Nfine*dt)),lambda0) for n in multi_N]      #
else:                                                                           #
    obs = np.matmul(splin.expm(C*(N*Nfine*dt)),lambda0)                         #
#-------------------------------------------------------------------------------#

# Write the minimisation problem as a linear system (A * X = Amatrix * X = A^T Y) with A = (F^N)^T F^N and b = (F^N)^T Y
if multi_obs:
    b=np.zeros(C.shape[0])
    for i in range(len(multi_N)):
        b = b+ np.dot(np.transpose(np.linalg.matrix_power(F,multi_N[i])),obs[i])
    b = b/N_obs
else:
    b = np.dot(np.transpose(np.linalg.matrix_power(F, N)),obs)


# Matrix in consideration, here A = (F^N)T*F^N
if multi_obs:
    A = np.zeros(C.shape[0])
    for i in range(len(multi_N)):
        A = A + np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),np.linalg.matrix_power(F,multi_N[i]))
    A = (1/len(multi_N))*A
else:
    A = np.matmul(np.transpose(np.linalg.matrix_power(F,N)),np.linalg.matrix_power(F,N))

if regularisation:
    if multi_obs:
        A = A + alpharegul*np.identity(A.shape[0])
    else:
        A = A + alpharegul*np.identity(A.shape[0])
        
        
l0,v = np.linalg.eig(A)
l1=np.abs(l0)
print('Conditioning of A matrix = ',np.max(l1)/np.min(l1),np.max(l1),np.min(l1))

# Inverse of the matrix, in practical applications not feasible to calculate
A_inv = np.linalg.inv(A)

# solution for the system, x* = A^-1 * b
x_star = np.matmul(A_inv,b)



# Run exact CG
lambda1=np.zeros(C.shape[0],dtype=A.dtype)
lambda1[:]=0.

# print('Launch Conjugate gradient')
lambdasol = conjgrad(Amatrix, F, G, N, b, lambda1, eps = 1.e-4)
# lambdasol = conjgrad(APmatrix, F, G, N, b, lambda1, epspara = 1.e-6, eps = 1.e-4)
niter = lambdasol[1]
print('Niter CG = ',niter)

# Compute the residual and its A-1 norm
r = np.dot(A,lambdasol[0])-b
rnorm=ep_norm(A_inv,r)
eps = (2.*ep_norm(A_inv,r)/ep_norm(A_inv,b))**2

# Run inexact CG
lambda1[:]=0.
print('Launch Inexact Conjugate gradient with eps = ',eps)
solutions = inexact_CG(b,eps=eps,x=lambda1,nitermax=2*niter)

print('\n########## Parameters ##########\n')

print('omegamax', dt*np.max(np.abs(l0)) )
print('q = {}'.format(Nfine/q))
print('theta = {}'.format(theta))
print('Number of time windows = {}'.format(N))
print('Number of minimisation iterations = {}'.format(solutions[1]))
print('Number of parareal iterations = {}'.format(sum(solutions[8])))


pf.plot(solutions)

pf.initial_states(nx,lambda0,lambdasol,solutions)


icgsol = Finesolution(F,solutions[0],N)
refsol = Finesolution(F,lambda0,N)

pf.contourplots(F,N,Dt,L,nx,icgsol,refsol)
