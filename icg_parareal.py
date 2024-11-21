#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:48:20 2023

"""

import numpy as np
import scipy.linalg as splin
import math
import time
from matplotlib.ticker import MaxNLocator

import matplotlib as mpl 

import matplotlib.pyplot as plt
from matplotlib import animation

# colormap = plt.get_cmap('Set2')
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[colormap(i) for i in np.arange(0,10)])

########## Plot pararmeters ###################
plt.rcParams.update({'font.size': 25})        #
plt.rcParams['lines.linewidth'] = 3.5         #
plt.rc('axes',xmargin=0.02)    #
plt.rc('xtick.major',size=7, width=2)      #
plt.rc('ytick.major',size=7, width=2)      #
plt.rc('xtick', direction ='inout')           #
plt.rc('ytick', direction ='inout')           #
plt.rc('figure',figsize=(8,6))                #
###############################################

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)




################################# Parareal ##################################

def Parareal(F,G,lambda0,N,k=None,eps=None,guess=None):
    
    if guess is not None:
        maxind = 1
    else:
        maxind = N
    lambdank = np.zeros((N+1,maxind+1,lambda0.shape[0]),dtype = type(lambda0))
    
    Flambdank = np.zeros(((N+1)*maxind+1,lambda0.shape[0]),dtype = type(lambda0))
    lambdank_s = np.zeros(((N+1)*maxind+1,lambda0.shape[0]),dtype = type(lambda0))

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
    nbvect = 0
    nbvect2 = 0
    for p in range(1,maxind+1):
        # print('Iteration: {}'.format(p))
        lambdank[0,p] = lambda0
        for n in range(1,N+1):
            Flambdank[nbvect,:]=np.dot(F,lambdank[n-1,p-1])
            lambdank_s[nbvect,:]=lambdank[n-1,p-1].copy()
            nbvect = nbvect + 1

        # Krylov enhanced parareal method
        if (Krylov_Enhanced):
            toto=lambdank_s[0:nbvect,:].astype(float)
            Q,R=np.linalg.qr(np.transpose(toto))
            Pk=np.matmul(Q,np.matmul(np.linalg.inv(np.matmul(np.transpose(Q),Q)),np.transpose(Q)))
            print('Projection rank : ',np.linalg.matrix_rank(Pk))

        for n in range(1,N+1):
            if (Krylov_Enhanced):
                PU=np.dot(Pk,lambdank[n-1,p])
                lambdank[n,p] = np.dot(F,PU) + np.dot(G,lambdank[n-1,p]-PU)
            else:
                lambdank[n,p] = Flambdank[nbvect2,:] - np.dot(G,lambdank[n-1,p-1]) + np.dot(G,lambdank[n-1,p])
            nbvect2=nbvect2+1

        if eps != None:
            err = np.linalg.norm(lambdank[N,p]-lambdank[N,p-1])
            print('Error: {}'.format(err))
            if err <= eps:
                break
        
    return lambdank,p


def Finesolution(F,lambda0,N):
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



''' Matrices associated with data assimilation problem '''

#### Error matrix ####
def Ematrix(F,G,N,k):
    if multi_obs:
        res = np.zeros(A.shape[0])
        for i in range(len(multi_N)):
            res = res - np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),Pararealerror(F,G,multi_N[i],k))
    else:
        res = -np.matmul(np.transpose(np.linalg.matrix_power(F,N)),Pararealerror(F,G,N,k))          

    return res


'''# Matrix of associated system Amatrix = (F^N)^T * F^N #'''
# Return Amatrix*x
def Amatrix(F,G,N,x,k=None,eps=None):
    if multi_obs:
        res = np.zeros(A.shape[0])
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


'''#A matrix of the associated linear system when one uses Parareal for the forward model#'''
'''# and the true model F for the backward model. #'''
# APmatrix = (F^N)^T * Parareal
# Return APmatrix*x
def APmatrix(F,G,N,x,k=None,eps=None):
    par,it = Parareal(F,G,x,N,k,eps)
    print('Iterations: ',it)
    
    if multi_obs:
        res = np.zeros(A.shape[0])
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



####################### Conjugate Gradient method ############################

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
    q_diff = []     # |q(x_k) - q(x_*)|
    res = []        # A-1 norm of residual
    res2 = []       # 2-norm of residual
    Ep = []         # ||Ep||_A^-1 values
    para_it = []    # number of parareal iterations
    q = []
    cf =[]
    
    r = b -A(F,G,N,x,k,epspara)
    pvec = [r]      # direction vector p
    
    beta_old = np.square(np.linalg.norm(b,2))
    
    reorth = True # Reorthogonalization or not
    u=np.zeros((2*N,x.shape[0]))
    u[0,:] = b/beta_old
    
    p = r
    rsold = np.dot(np.transpose(r),r)

    iteration = 0
    while True:
        
        print("########## CG iteration {} ###########".format(iteration))
        Ap = A(F,G,N,p,k,epspara)
        
        #ap,it = Parareal(F,G,p,N,eps=1.e-6)
        #ap1 = ap[N,it]
        #Ap = np.matmul(np.transpose(np.linalg.matrix_power(F,N)),ap1)
        #para_it.append(it)
        
        # To check with accurate parareal
        # sol,Niter = Parareal(F,G,p,N,eps=1.e-10)
        # e = []
        # k = 0
        # while k <Niter:
        #     e.append(np.linalg.norm(sol[N,k]-sol[N,Niter],2))
        #     k = k+1
        
        # Ep.append(e)
        # para_it.append(Parareal(F,G,p,N,eps=epspara)[1])
        
        
        tol = 0.5*np.sqrt(eps)*ep_norm(M_inv,b)         # icg tolerance for A^-1 norm
        qtol = eps*np.abs(quad(x_star))                 # icg tolerance for q_diff
        
        alpha = rsold/np.dot(np.transpose(p),Ap)
        x = x + np.dot(alpha,p)
        r = r -np.dot(alpha,Ap)
        rsnew = np.dot(np.transpose(r),r)
        
        # storing quantities for comparing
        q.append(quad(x))
        cf.append(cost_function(F,N,x,obs))
        q_diff.append(np.abs(quad(x) - quad(x_star)))
        res.append(ep_norm(M_inv,r))
        res2.append(np.sqrt(rsnew))
        
        # print residual norms
        print('2-norm of r: ', res2[iteration])
        print('A-1 norm of r: ', res[iteration],'\n')
        
        if np.sqrt(rsnew) <eps:
            break
        
        if reorth:
         	for i in range(iteration+1):
                  r = r -np.dot(u[i,:],r)*u[i,:]
         	beta_new = np.matmul(np.transpose(r),r)
         	u[iteration+1,:]=r/np.sqrt(beta_new)
        else:
            beta_new = np.matmul(np.transpose(r),r)
        
        # p = r +(rsnew/rsold)*p
        p = r + (beta_new/beta_old)*p
        rsold = rsnew

        beta_old = beta_new
        
        pvec.append(p)
        iteration+=1
    return x, iteration+1, q_diff, res, res2, para_it, tol, qtol, pvec,Ep, q, cf



def plotcg(cg):
    
    it = cg[1]
    n = np.arange(0,it)
    qd = cg[2]
    res = cg[3]
    res2 = cg[4]
    para_it = cg[5]
    tol = cg[6]
    qtol = cg[7]
    
    ###### for APmatrix ######
    #fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8))
    #ax1.tick_params(bottom=True, top=True, left=True, right=True)
    #ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    #plt.margins(x=0.01)
    #ax1.xaxis.set_ticks([0,5,10,15,20,25])
    #ax1.set_yscale('log')
    #ax1.set_ylabel(r'$\Vert \mathbf{r}_j \Vert_2$',fontsize=30)
    #ax1.hlines(1.e-4,0,25,'k',linestyle='dashed',label=r'$\epsilon_{\rm cg}$')
    ## ax1.plot(n,res,'r',label=r'$\Vert \mathbf{r}_j \Vert_{\mathbf{A}^{-1}}$')
    #ax1.plot(n,res2,'b',label=r'$\Vert \mathbf{r}_j \Vert_2$')
    #ax1.set_xlabel(r'CG iteration $j$',fontsize=30, labelpad=10)
    #
    #ax2.set_ylim(6,11)
    #ax2.set_ylabel(r'parareal iterations $k$',fontsize=30)
    #ax2.plot(n,para_it,'o--')
    #ax2.xaxis.set_ticks([0,5,10,15,20,25])
    #ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax2.set_xlabel(r'CG iteration $j$',fontsize=30, labelpad=10)
    #fig.tight_layout()
    #
    #plt.savefig('figure.pdf',format = 'pdf',dpi=1000)
    
    
    ##### for exact matrix ####
    f, ax = plt.subplots(1,1,figsize=(12,10))
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    ax.set_xticks(ticks=np.arange(0,30,5))
    ax.set_yscale('log')
    ax.set_xlabel(r'CG iteration $j$',fontsize=30, labelpad=10)
    ax.set_ylabel(r'$\Vert \mathbf{r}(\mathbf{x}_j) \Vert_2$',fontsize=30, labelpad=10)
    ax.hlines(1.e-4,0,25,'k',linestyle='dashed',label=r'$\epsilon_{\rm cg}$')
    ax.plot(n,res2,'b',label=r'$\Vert \mathbf{r}(\mathbf{x}_j) \Vert_2$')
    f.tight_layout()

    plt.savefig('figure.pdf',format = 'pdf',dpi=1000)



############### Data assimilation and minimisation #################

'''# regularisation based on spatial derivative of the initial state #'''

##########################
regularisation = True    #
alpharegul = 1.e-5       #
##########################

def penalisation(lambda0):
    res = 0
    for i in range(len(lambda0)-1):
        res = res + 0.5*alpharegul*(((lambda0[i+1] - lambda0[i])/dx)**2)
    return res

def pen_matrix():
    mat = np.zeros_like(A)
    mat[len(mat)-1,len(mat)-1] = 2
    for i in range(len(lambda0)-1):
        mat[i,i] = 2
        mat[i,i+1] = -1
        mat[i+1,i] = -1
    # mat[len(mat)-1,len(mat)-1] = -1
    # for i in range(len(lambda0)-1):
    #     mat[i,i] = -1
    #     mat[i,i+1] = 1
    return -mat*(1/dx**2)


'''# Cost function for data assmiliation, J(x) = 0.5*|| Ax - Y||^2   Y = obs #''' 
def cost_function(F,N,lambda0,obs):
    if multi_obs:
        cf = 0
        for i in range(len(multi_N)):
            cf = cf + (1/2*N_obs)*np.square((np.linalg.norm(np.matmul(np.linalg.matrix_power(F,multi_N[i]),lambda0) - obs[i],2)))
        cf = cf/(len(multi_N))
    else:
        cf = 0.5*np.square((np.linalg.norm(np.matmul(np.linalg.matrix_power(F, N),lambda0)-obs,2)))
        
    if regularisation:
        if multi_obs:
            cf = cf + 0.5*alpharegul*np.square(np.linalg.norm(lambda0,2))
        else:
            cf = cf + 0.5*alpharegul*np.square(np.linalg.norm(lambda0,2))
    return cf


'''# Gradient/Residual for Ax = b,   Del J(x) = (A^T)*(Ax - Y) #'''
def residual(x):
    grad = np.matmul(M,x) - b
    return grad

# ellipsoidal norm
def ep_norm(P,x):
    return np.sqrt(np.matmul(np.transpose(x),np.matmul(P,x)))

# quadratic form, q(x) = 0.5*(x^T M x) - b^T x
def quad(x):
    q1 = 0.5*(np.matmul(np.transpose(x),np.matmul(M,x))) - np.matmul(np.transpose(b),x)
    return q1

# bound for E_norm
def E_bound(eps,b,p,r,phi):
    eps1 = np.sqrt(eps)
    b1 = ep_norm(M_inv,b)
    p1 = ep_norm(M,p)  
    r1 = np.square(np.linalg.norm(r,2))
    
    omega = (eps1*b1*p1)/(2*phi*r1 + eps1*b1*p1)
    return omega

# Primal dual matrix norm for perturbation
def E_norm(B,k):
    P = splin.sqrtm(B)
    P_inv = np.linalg.inv(P)
        
    E = np.matmul(np.matmul(P_inv,Ematrix(F,G,N,k)),P_inv)   
   
    return np.linalg.norm(E,2)     

# approximations from Gratton et al.

def prac_Enorm(E):
    lam = 1/min(np.abs(np.linalg.eig(M)[0]))
    return lam*np.linalg.norm(E,2)

def prac_pnorm(p):
    return np.sqrt(np.trace(M)/M.shape[0])*np.linalg.norm(p,2)

def inexact_quad(x):
    return -0.5*np.matmul(np.transpose(b),x)

def prac_bnorm(x,it):
    if it == 0:
        return np.linalg.norm(b,2)/np.sqrt(max(np.abs(np.linalg.eig(M)[0])))
    else:
        return np.sqrt(2*abs(inexact_quad(x)))
    
def prac_Ebound(eps,x,p,r,phi,it):
    
    eps1 = np.sqrt(eps)
    if it == 0:
        q1 = (np.sqrt(2)*np.linalg.norm(b,2))/(np.sqrt(max(np.abs(np.linalg.eig(M)[0]))))
    else:
        q1 = np.sqrt(np.abs(inexact_quad(x)))
    t1 = np.sqrt(np.trace(M))
    p1 = np.linalg.norm(p,2)
    r1 = np.square(np.linalg.norm(r,2))
    
    return (eps1*q1*t1*p1)/(np.sqrt(2*M.shape[0])*phi*r1 + eps1*q1*t1*p1)

def obs_error(p1,k):
    a = []
    par,it = Parareal(F,G,p1,N,k,eps)

    for i in range(len(multi_N)):
        a1 = np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),par[multi_N[i],it])
        a2 = np.dot(np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),np.linalg.matrix_power(F,multi_N[i])),p1)
        
        a.append(ep_norm(M_inv,a1-a2))
        
        
    return a

# cost function if multiple observations are used
def multi_cf(F,lambda0,i,obs):
    cf=0
    cf = cf + (1/2)*np.square((np.linalg.norm(np.matmul(np.linalg.matrix_power(F,multi_N[i]),lambda0) - obs[i],2)))
    if regularisation:
        cf = cf + 0.5*alpharegul*np.square(np.linalg.norm(lambda0,2))
        
    return cf


##################### Inexact Conjugate Gradient ###########################
def inexact_CG(b,eps,x,nitermax):
    
    qd = []           # quadratic difference
    gap1 = []         # residual gap
    res = [-b]        # inexact residual
    res_norm = []     # inexact residual norm
    e_bd = []         # omega
    para_iter = []    # parareal iteration array
    pvec = [b]        # direction vector
    e_bd2 = []        # omega2    
    cg_sol=[x]        # icg iterates
    cf = [cost_function(F,N,np.zeros(lambda0.shape[0]),obs)]    #cost function
    grad = [np.linalg.norm(residual(np.zeros(lambda0.shape[0])))]   # gradient
    para_tol = []
    
    beta_old = np.square(np.linalg.norm(b,2))
    r = -b
    p = b
    
    ######### change the integer for the stopping criterion##########
    d = 2                                                           #
    #################################################################
    
    q = [0]     # quadratic values
    
    bound = 0.5*np.sqrt(eps)*ep_norm(M_inv, b)      # theoretical tolerance
    print('Minimisation tolerance ',bound )
    iteration = 0    
    
    phi = nitermax
    big_phi = 1
    print('nitermax = ',nitermax)

    
    reorth = True # Reorthogonalization or not
    u=np.zeros((nitermax,x.shape[0]))
    u[0,:] = b/beta_old
        
    # testing orthogonality condition
    # sol_array = np.zeros((nitermax,x.shape[0]))
    ep_diff = []
    p_diff = []
    mult_cf = []
   
    ob_err1 = np.zeros((nitermax,len(multi_N)))
    while True:
        print('#########################################')
        print('Iteration {} '.format(iteration),'\n')
        
        inacc_budget = True
        
        print('phi : ', phi)
        
        #omega = E_bound(eps,b,p,r,phi)
        omega = prac_Ebound(eps,x,p,r,phi,iteration)
        print("omega: ", omega)
        e_bd.append(omega)
        
        #omega2 = omega*ep_norm(M,p)
        omega2 = omega*prac_pnorm(p)
        print("omega x ||p||: ",omega2)
        e_bd2.append(omega2)

        
        # with || E ||_A^{-1},A 
        
        #k2 = 1
        #while k2<N:
        #    e = E_norm(M,k2)
        #    print('Error at parareal iteration ',k2,' : ',e)
        #    if (e>omega and k2!=N-1):
        #        k2 = k2+1
        #       
        #    elif k2==N-1:
        #        print('Number of parareal iterations : ',k2)
        #        omega_hat = E_norm(M,k2)
        #        print('omega_hat : ',omega_hat)
        #        para_iter.append(k2)
        #       
        #        c = APmatrix(F,G,N,p,k2)
        #        break
        #       
        #    else:
        #        print('Number of parareal iterations : ',k2)
        #        omega_hat = E_norm(M,k2)
        #        print('omega_hat : ',omega_hat)
        #        para_iter.append(k2)
        #       
        #        c = APmatrix(F,G,N,p,k2)
        #        break

        

        # with exact || Ep ||_A^{-1}

        #k1=1
        #store = []
        #while k1<=N:
        #    # e = ep_norm(M_inv,APmatrix(F,G,N,p,k1) - np.dot(M,p))
        #    # e = ep_norm(M_inv,np.dot(Ematrix(F,G,N,k1),p))
        #    e = ep_norm(M_inv,APmatrix(F,G,N,p,k1)-Amatrix(F,G,N,p))
        #    print('|| Ep || at parareal iteration ',k1,' : ',e)
        #              
        #    # store.append(e)
        #    if e > omega2:
        #        k1 = k1+1
        #    else:
        #        print('Number of parareal iterations : ',k1)
        #        # omega_hat = ep_norm(M_inv,np.dot(Ematrix(F,G,N,k1),p))
        #        omega_hat = e
        #        print('omega_hat: ',omega_hat)
        #        para_iter.append(k1)
        #      
        #        ob_err = obs_error(p,k1)
        #        ob_err1[iteration] = ob_err

        #        c = APmatrix(F,G,N,p,k1)
        #        break
               


        # using last parareal iterate as approximation for || Ep ||_A^{-1}
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
                para_tol.append(omega_hat)
                print('omega hat : ', omega_hat)
                print('parareal iterations ', i+1)
                ksol=i+1
                para_iter.append(ksol)
                break
        p_diff.append(p_vals)
        ep_diff.append(ep_vals)
        print('ksol for c calculation', ksol)

        if multi_obs:
            c = np.zeros(A.shape[0])
            for i in range(len(multi_N)):
                c = c + np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),x_est[multi_N[i],ksol])
        else:
            c = np.matmul(np.transpose(np.linalg.matrix_power(F,N)),x_est[N,ksol])
            # c = np.matmul(np.transpose(np.linalg.matrix_power(F,N)),sol_new)
            
        if regularisation:
            if multi_obs:
                c = c + alpharegul*np.matmul(pen_matrix(),p)
            else:
                c = c + alpharegul*p


        if inacc_budget:
            #phi_hat = ((ep_norm(M,p)-omega_hat)*np.sqrt(eps)*ep_norm(M_inv,b)*ep_norm(M,p))/(omega_hat*2*np.square(np.linalg.norm(r,2)))
            phi_hat = ((prac_pnorm(p)-omega_hat)*np.sqrt(eps)*prac_bnorm(x,iteration)*prac_pnorm(p))/(omega_hat*2*np.square(np.linalg.norm(r,2)))
            print('phi : ',phi,'\nphi_hat : ',phi_hat,'\nphi_hat > phi : ',phi_hat>phi)
            big_phi_new = big_phi - (1/phi_hat)
            
            if iteration<nitermax:
                phi_new = (nitermax-iteration-1)/big_phi_new
            else:
                phi_new = phi
                
            phi = phi_new
            big_phi = big_phi_new


        res_gap = ep_norm(M_inv,residual(x)-r)
        gap1.append(res_gap)

        # bound for quadratic
        quad_bound = eps*np.abs(quad(x_star))
        
        alpha = beta_old/np.matmul(np.transpose(p),c)
        x_new = x + np.dot(alpha,p)
        r_new = r + np.dot(alpha,c)
        
        q.append(inexact_quad(x_new))
        cg_sol.append(x_new)
        cf.append(cost_function(F,N,x_new,obs))
        grad.append(np.linalg.norm(residual(x_new)))
        
        mcf=[]
        for j in range(N_obs):
            mcf.append(multi_cf(F,x_new,j,obs))
            
        mult_cf.append(mcf)
               
        quad_diff = np.abs(quad(x_new)-quad(x_star))
        qd.append(quad_diff)
        print('2-norm of r: ', np.linalg.norm(r_new))
        print('A-1 norm of r: ', ep_norm(M_inv,r_new), '\n')
        
        res_norm.append(ep_norm(M_inv,r_new))
        
        #### inexact CG stopping criterion####

        # stoppiing criterion using ||r_j||_A^{-1}
        #if ep_norm(M_inv,r_new) <= bound:
        #    break

        # using approximations
        if iteration >=d:
            if (q[iteration +1 -d] - q[iteration+1]) <= 0.25*eps*np.abs(inexact_quad(x_new)):
                 break

        # reorthogonalisation
        if reorth:
        	for i in range(iteration+1):
        		r_new = r_new -np.dot(u[i,:],r_new)*u[i,:]
        	beta_new = np.matmul(np.transpose(r_new),r_new)
        	u[iteration+1,:]=r_new/np.sqrt(beta_new)
        else:
            beta_new = np.matmul(np.transpose(r_new),r_new)
            
        p_new = -r_new + np.dot((beta_new/beta_old),p)
        
        ####update old values####
        beta_old = beta_new
        r = r_new
        x = x_new
        p = p_new
        
        pvec.append(p)
        res.append(r)
        
        iteration+=1
            
    return x_new, iteration+1,res_norm, qd, bound, quad_bound, gap1, e_bd, para_iter,pvec,e_bd2,res,q,cg_sol,cf,grad,para_tol,p_diff,ep_diff, ob_err1, mult_cf


# plotting the results of inexact_CG
def plot(cg):    
    # unpacking values
    it = cg[1]   
    res = cg[2]   
    q_diff = cg[3]
    tol = cg[4]
    quad_tol = cg[5]
    gap1 = cg[6]
    e_bd = cg[7]
    e_bd2 = cg[10]
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8))
    
    plt.margins(x=0.01)
    ax1.set_yscale('log')
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    
    ax1.set_xlabel(r'icg iteration $j$',fontsize=30, labelpad=10)

    ax1.hlines(tol,0,it+1,'k',linestyle = 'dotted', lw=1.5)
    ax1.annotate(r'$\frac{1}{2}\sqrt{\epsilon_{\rm icg}} \,\Vert \mathbf{b} \Vert_{\mathbf{A}^{-1}}$',xy=(12,tol*0.3),fontsize=22)

    ax1.hlines(quad_tol,0,it+1,'k',ls=(0,(5,5)),lw=1.5)
    ax1.annotate(r'$\epsilon_{\rm icg} \,|\,J(\mathbf{x}_\ast)|$',xy=(10,quad_tol*0.3),fontsize=22)

    ax1.plot(np.arange(0,it-2),q_diff[:-2],'brown',ls=(0,(5,1)))
    ax1.annotate(r'$|\, J(\mathbf{x}_j) - J(\mathbf{x}_\ast)|$',xy=(18,2.3e-6),fontsize=22,c='brown')
    
    ax1.plot(np.arange(0,it-2),res[:-2],'r',ls=(0,(3,2,1,2)))
    ax1.annotate(r'$\Vert \mathbf{r}_j \Vert_{\mathbf{A}^{-1}}$',xy=(21,1.8e-3),fontsize=22,c='r')

    ax1.plot(np.arange(0,it-2),e_bd[:-2],'m',ls=(0,(3,1,1,1,1,1)))
    ax1.annotate(r'$\omega_j$',xy=(23,0.025),fontsize=22,c='m')

    
    #ax1.plot(np.arange(0,it),e_bd2,c='orange',ls=(30,(3,2,1,2,1,2)))
    #ax1.annotate(r'$\xi_j$',xy=(23.2,5.9e-6),fontsize=22,c='orange')    
    
    ax1.plot(np.arange(0,it-2),gap1[:-2],'g',ls=(0,(1,3)))
    ax1.annotate(r'$\Vert \mathbf{r}(\mathbf{x}_j)-\mathbf{r}_j \Vert_{\mathbf{A}^{-1}}$',xy=(18,2e-5),fontsize=22,c='g')

    ax1.xaxis.set_ticks(np.arange(0,26,5))
    # ax1.legend(fontsize=17,loc='lower right')
    
    ax2.plot(cg[8][:-2],'o--')
    ax2.set_ylim(4,9)
    ax2.xaxis.set_ticks(np.arange(0,30,5))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel(r'icg iteration $j$',fontsize=30, labelpad=10)
    ax2.set_ylabel(r'parareal iterations $k$',fontsize=30, labelpad=10)
    
    fig.tight_layout()
    
    plt.savefig('figure.pdf',format = 'pdf',dpi=1000)
    
    
####################### Analysis ##########################################
Krylov_Enhanced = False
######################## Parareal parameters ##############################   
omegamax = 0.29997 # Courant number                                          #
N = 20  # Set N number of time windows                                    #
                                                                          #
# Compute a dt such that max(|eigenvalues(A)|)*dt = omegamax              #
# dt = omegamax/np.max(np.abs(l0))                                        #
# Dt = dt * Nfine / q                                                     #
                                                                          #
# q has to divide Nfine                                                   #
dt = 0.05       # fine time-step                                          #
Nfine = 100    # no. of fine time steps per time windows                  #
q=20            # no. of coarse time steps per time windows               #
Dt = dt*Nfine/q  # coarse time-step                                       #
theta = 0.51                                                              #
###########################################################################


######### Shallow water model parameters #############
H=0.9        # mean height                           #
g=10.       # gravity                               #
L=120.       # horizontal length                     #
nx = 120     # no. of spatial  grid points           #
dx=L/nx      # spatial resolution                    #
###################################################### 


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

visc = 0.15
print("\n ------Diffusion courant number------", (visc*dt)/(dx**2),'\n')
A=np.zeros((2*nx-1,2*nx-1))

# d eta/dt = -H * du/dx
for i in range(nx):
    if (i==0):
        A[i,nx+i]=-H/dx
    elif (i==nx-1):
        A[i,nx+i-1]=H/dx
    else:
        A[i,nx+i]=-H/dx
        A[i,nx+i-1]=H/dx

for i in range(nx-1):
    A[nx+i,i+1]=-g/dx
    A[nx+i,i]=g/dx
    if (i!=0):
        A[nx+i,nx+i-1] = visc/dx**2
    if (i!=nx-2):
        A[nx+i,nx+i+1] = visc/dx**2
    A[nx+i,nx+i] = (-2*visc)/dx**2
    

# xr(0:nx-1) : locations of eta points
xr=np.zeros(nx)
for i in range(nx):
    xr[i]=i*dx+dx/2.

l0,v = np.linalg.eig(A)
l1 = np.abs(l0)
print('Conditioning of A matrix = ',np.max(l1)/np.min(l1))

print(0.05*np.max(l1))

F = np.linalg.matrix_power(np.matmul(np.linalg.inv(np.identity(A.shape[0])-theta*(dt)*A),np.identity(A.shape[0])+(1.-theta)*(dt)*A),Nfine)
G = np.linalg.matrix_power(np.matmul(np.linalg.inv(np.identity(A.shape[0])-theta*(Dt)*A),np.identity(A.shape[0])+(1.-theta)*(Dt)*A),q)

# propagator norms
print('||F||_2 = {}, ||G||_2 = {}, \n||F-G||_2 = {}'.format(np.linalg.norm(F,2),np.linalg.norm(G,2),np.linalg.norm(F-G,2)),'\n')
print('Total integration time = ',dt*Nfine*N)


lambda0=np.zeros(A.shape[0],dtype=A.dtype)
# Initialize a Gaussian at the middle of the basin
lambda0[:]=0.
for i in range(nx):
    lambda0[i]=np.exp(-((xr[i]-L/2.)/(L/15.))**2) #*np.sin(20.*xr[i]/L)


######### reference solution ############
refsol=Finesolution(F,lambda0,N)[N]

# refsol from parareal #

lambdank,it = Parareal(F,G,lambda0,N,eps=1.e-6)
print('Iterations',it)

# plot solutions
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (12,6))
# free surface
ax1.plot(xr,lambdank[N,it][0:nx])
ax1.set_title('free surface')

# velocity
ax2.plot(lambdank[N,it][nx:])
ax2.set_title('velocity')
plt.show()


####################### Analysis for inexact CG #############################

#################### use single/multiple observations #######################
multi_obs = False                                                            #
#time windows where observations are defined                                #
# multi_N = [10*i for i in range(1,3)]                                      #
multi_N = [4*i for i in range(1,6)]

N_obs = len(multi_N)
                                                                            #
# True observation at the end of integration time                           #
# if multi_obs:                                                             #
#     obs = [Finesolution(F,lambda0,n)[n] for n in multi_N]                 #    
# else:                                                                     #    
#     obs = Finesolution(F,lambda0,N)[N]                                    #    
                                                                            #        
#### OR using the exact solution ####                                       #    
if multi_obs:                                                               #
    obs = [np.matmul(splin.expm(A*(n*Nfine*dt)),lambda0) for n in multi_N]  #
else:                                                                       #
    obs = np.matmul(splin.expm(A*(N*Nfine*dt)),lambda0)                     #
#############################################################################

'''# Write the minimisation problem as a linear system (A^TA * X = Amatrix * X = b = A^T Y) with A = F^N #'''
if multi_obs:
    b=np.zeros(A.shape[0])
    for i in range(len(multi_N)):
        b = b+ np.dot(np.transpose(np.linalg.matrix_power(F,multi_N[i])),obs[i])
    b = b/N_obs
else:
    b = np.dot(np.transpose(np.linalg.matrix_power(F, N)),obs)


'''# Matrix in consideration, here M = A = (F^N)T*F^N #'''
if multi_obs:
    M = np.zeros(A.shape[0])
    for i in range(len(multi_N)):
        M = M + np.matmul(np.transpose(np.linalg.matrix_power(F,multi_N[i])),np.linalg.matrix_power(F,multi_N[i]))
    M = (1/len(multi_N))*M
else:
    M = np.matmul(np.transpose(np.linalg.matrix_power(F,N)),np.linalg.matrix_power(F,N))

if regularisation:
    if multi_obs:
        M = M + alpharegul*np.identity(M.shape[0])
    else:
        M = M + alpharegul*np.identity(M.shape[0])
        
        
l0,v = np.linalg.eig(M)
l1=np.abs(l0)
print('Conditioning of M matrix = ',np.max(l1)/np.min(l1),np.max(l1),np.min(l1))

# Inverse of the matrix, in practical applications not feasible to calculate
M_inv = np.linalg.inv(M)

# solution for the system, x* = A^-1 * b
x_star = np.matmul(M_inv,b)

# exact CG
lambda1=np.zeros(A.shape[0],dtype=A.dtype)
lambda1[:]=0.

# print('Launch Conjugate gradient')
lambdasol = conjgrad(Amatrix, F, G, N, b, lambda1, eps = 1.e-4)
# lambdasol = conjgrad(APmatrix, F, G, N, b, lambda1, epspara = 1.e-6, eps = 1.e-4)
niter = lambdasol[1]
print('Niter CG = ',niter)

# Compute the residual and its A-1 norm
r = np.dot(M,lambdasol[0])-b

rnorm=ep_norm(M_inv,r)
eps = (2.*ep_norm(M_inv,r)/ep_norm(M_inv,b))**2

# inexact CG
ts = time.time()

lambda1[:]=0.
print('Launch Inexact Conjugate gradient with eps = ',eps)
solutions=inexact_CG(b,eps=eps,x=lambda1,nitermax=2*niter)

# print('Time taken : ', time.time()-ts)
print('\n########## Parameters ##########\n')

print('omegamax', dt*np.max(np.abs(l0)) )
print('q = {}'.format(Nfine/q))
print('theta = {}'.format(theta))
print('Number of time windows = {}'.format(N))
print('Number of minimisation iterations = {}'.format(solutions[1]))
print('Number of parareal iterations = {}'.format(sum(solutions[8])))



#################### functions for plots ###############################

# ||E(k)||_A^-1,A exact and approx norms plot
def Eplot():
    e = []
    e_approx =[]
    for i in range(N):
        e.append(E_norm(M,i))
        e_approx.append(prac_Enorm(Ematrix(F,G,N,i)))
        
    plt.figure(0,figsize=(8,6))
    plt.yscale('log')
    plt.hlines(1,0,N,'k',ls='dashed')
    plt.plot(np.arange(0,N),e,'o--',label='exact')
    plt.plot(np.arange(0,N),e_approx,'+--',label='approx')
    plt.legend()
    plt.title('Exact and approx Enorm')
    


# ||Ep||_A^-1 plots
def ep_plot(i):
    
    p = solutions[9][i]
    ep = []
    for i in range(N):
        ep.append(ep_norm(M_inv,APmatrix(F,G,N,p,k=i)-np.dot(M,p)))
                  
    omega2 = solutions[10][i]
    
    plt.figure(0,figsize=(12,10))
    plt.yscale('log')
    # plt.ylim(1.e-12,1.e-3)
    plt.hlines(omega2,0,N,'k',ls='dashed',label=r'$\xi_j$',lw=2)
    plt.plot(np.arange(0,N),ep,'bo',ms=12,mew=4,mfc='none',lw=4,label=r'$\Vert E_j\textbf{p}_j \Vert_{A^{-1}}$')
    # 5th iteration
    plt.plot(5,ep[5],'ro',ms=12,mew=4,mfc='none',lw=4)
    plt.vlines(5,0,ep[5],'r',ls='dotted')
    # 6th iteration
    plt.plot(6,ep[6],'ro',ms=12,mew=4,mfc='none',lw=4)
    plt.vlines(6,0,ep[6],'r',ls='dotted')
    plt.title('Modified Criterion')

    plt.legend()
    
   
# solutions from exact and inexact CG     
def results():

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8),sharey= True)
    
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax1.hlines(1.,0,121,'w')
    ax1.set_xticks(np.arange(0,121,30))
    ax1.plot(refsol[0:nx],'r',ls=(0,(1,1,1,1,1)),label='Reference')
    ax1.plot(Finesolution(F,lambdasol[0],N)[N][0:nx],'b-.',lw=2,label='CG')
    ax1.plot(Finesolution(F,solutions[0],N)[N][0:nx],'m--',lw=2,label='Inexact CG')
    ax1.set_title(r'free surface $\eta$',fontsize=30)
    ax1.set_xlabel('$x$',fontsize=25)
    ax1.set_ylabel('values',fontsize=30)
    ax1.legend(fontsize=22)
    
    ax2.tick_params(bottom=True, top=True, left=True, right=True)
    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax2.hlines(1.,0,121,'w')
    ax2.set_xticks(np.arange(0,121,30))
    ax2.plot(refsol[nx:],'r',ls=(0,(1,1,1,1,1)),label='Reference')
    ax2.plot(Finesolution(F,lambdasol[0],N)[N][nx:],'b-.',lw=2,label='CG')
    ax2.plot(Finesolution(F,solutions[0],N)[N][nx:],'m--',lw=2,label='Inexact CG')
    ax2.set_title(r'velocity $u$',fontsize=30)
    ax2.set_xlabel('$x$',fontsize=30)
    ax2.legend(fontsize=22)
    fig.tight_layout()
    
    
    
# initial condition obtained from exact and inexact CG
def initial_states():
    f1, (a1,a2) = plt.subplots(1,2,figsize=(18,8),sharey=True)

    a1.margins(x=0.04)
    a1.tick_params(bottom=True, top=True, left=True, right=True)
    a1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    a1.hlines(1.,0,121,'w')
    a1.set_xticks(np.arange(0,121,30))
    a1.plot(lambda0[:nx],'r',ls=(0,(1,1,1,1,1)),label=r'Reference')
    a1.plot(lambdasol[0][:nx],'b-.',label='CG')
    a1.plot(solutions[0][:nx],'m--',label='Inexact CG')
    a1.set_ylabel('values',fontsize=30)
    a1.set_xlabel('$x$',fontsize=30)
    # a1.set_title(r'free surface $\eta_0(x)$')
    a1.legend(fontsize=22)

    a2.tick_params(bottom=True, top=True, left=True, right=True)
    a2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    a2.margins(x=0.025)
    a2.hlines(1.,0,121,'w')
    a2.set_xticks(np.arange(0,121,30))
    a2.plot(lambda0[nx:],'r',ls=(0,(1,1,1,1,1,)),label=r'Reference')
    a2.plot(lambdasol[0][nx:],'b-.',label='CG')
    a2.plot(solutions[0][nx:],'m--',label='Inexact CG')
    a2.set_xlabel('$x$',fontsize=30)
    # a2.set_title(r'veclocity $u_0(x)$')
    a2.legend(fontsize=22)
    f1.tight_layout()


    
def cf_grad():
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.set_yscale('log')
    ax.plot(solutions[14],label='cost function')
    ax.plot(solutions[15],label='gradient')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('values')
    
    ax.legend()



def contourplots(cgsol,refsol):
    
    t = np.linspace(0,N*Dt,N+1)
    t = np.array(t,dtype=float)
    
    x = np.linspace(0,L,int(L))
    x = np.array(x,dtype=float)
    
    X,T = np.meshgrid(x,t)
    
    Z = Finesolution(F,cgsol[0],N)[:,:nx]
    Z = np.array(Z,dtype=float)
    
    Z1 = Finesolution(F,refsol,N)[:,:nx]
    Z1 = np.array(Z1,dtype=float)
    
    f1,a1 = plt.subplots(figsize=(14,10),layout='tight')
    plt.contourf(X,T,Z,cmap='cividis')
    a1.set_xlabel('x')
    a1.set_ylabel('t')
    a1.set_title(r'Inexact CG solution: $\eta$')
    plt.colorbar()
    plt.savefig('contour_icg_eta.pdf',format = 'pdf',dpi=1000)

    f2,a2 = plt.subplots(figsize=(14,10),layout='tight')
    a2.set_xlabel('x')
    a2.set_ylabel('t')
    a2.set_title(r'Exact solution: $\eta$')
    plt.contourf(X,T,Z1,cmap='cividis')
    plt.colorbar()
    plt.savefig('contour_exact_eta.pdf',format = 'pdf',dpi=1000)

    

    
    

