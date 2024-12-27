import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl 
import matplotlib.pyplot as plt


#--------------- functions for plotting results -----------------#

#--------- Plot pararmeters ------------------#
plt.rcParams.update({'font.size': 25})        #
plt.rcParams['lines.linewidth'] = 3.5         #
plt.rc('axes',xmargin=0.02)                   #
plt.rc('xtick.major',size=7, width=2)         #
plt.rc('ytick.major',size=7, width=2)         #
plt.rc('xtick', direction ='inout')           #
plt.rc('ytick', direction ='inout')           #
plt.rc('figure',figsize=(8,6))                #
#---------------------------------------------#

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)


# Plotting the results of conjugate gradient
def plotcg_amat(cg):
    '''
    Plot when using Amatrix in conjgrad
    '''

    it = cg[1]
    n = np.arange(0,it)
    res = cg[2]
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8))
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    plt.margins(x=0.01)
    ax1.xaxis.set_ticks([0,5,10,15,20,25])
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$\Vert \mathbf{r}_j \Vert_2$',fontsize=30)
    ax1.hlines(1.e-4,0,25,'k',linestyle='dashed',label=r'$\epsilon_{\rm cg}$')
    ax1.plot(n,res2,'b',label=r'$\Vert \mathbf{r}_j \Vert_2$')
    ax1.set_xlabel(r'CG iteration $j$',fontsize=30, labelpad=10)
    
    ax2.set_ylim(6,11)
    ax2.set_ylabel(r'parareal iterations $k$',fontsize=30)
    ax2.plot(n,para_it,'o--')
    ax2.xaxis.set_ticks([0,5,10,15,20,25])
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel(r'CG iteration $j$',fontsize=30, labelpad=10)
    fig.tight_layout()
    
    plt.savefig('figure.pdf',format = 'pdf',dpi=1000)


def plotcg_apmat(cg):
    '''
    Plot when using APmatrix in conjgrad
    '''

    it = cg[1]
    n = np.arange(0,it)
    res = cg[2]
    para_it = cg[5]
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




# plotting the results of inexact conjugate gradient
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
    ax1.annotate(r'$\frac{1}{2}\sqrt{\epsilon_{\rm icg}} \,\Vert \mathbf{b} \Vert_{\mathbf{A}^{-1}}$',xy=(12,tol*0.4),fontsize=22)
    ax1.hlines(quad_tol,0,it+1,'k',ls=(0,(5,5)),lw=1.5)
    ax1.annotate(r'$\epsilon_{\rm icg} \,|\,J(\mathbf{x}_\ast)|$',xy=(4,quad_tol*0.4),fontsize=22)


    # uncomment if using || E ||_A^{-1},A
    #ax1.plot(np.arange(0,it),q_diff,'brown',ls=(0,(5,1)))
    #ax1.annotate(r'$|\, J(\mathbf{x}_j) - J(\mathbf{x}_\ast)|$',xy=(15,9e-8),fontsize=22,c='brown')
    #ax1.plot(np.arange(0,it),res,'r',ls=(0,(3,2,1,2)))
    #ax1.annotate(r'$\Vert \mathbf{r}_j \Vert_{\mathbf{A}^{-1}}$',xy=(21,1.8e-3),fontsize=22,c='r')
    #ax1.plot(np.arange(0,it),e_bd,'m',ls=(0,(3,1,1,1,1,1)))
    #ax1.annotate(r'$\omega_j$',xy=(23,0.025),fontsize=22,c='m')
    #ax1.plot(np.arange(0,it),gap1,'g',ls=(0,(1,3)))
    #ax1.annotate(r'$\Vert \mathbf{r}(\mathbf{x}_j)-\mathbf{r}_j \Vert_{\mathbf{A}^{-1}}$',xy=(19,1e-4),fontsize=22,c='g')
    #ax1.xaxis.set_ticks(np.arange(0,26,5))
    #ax2.plot(cg[8],'o--')
    #ax2.set_ylim(16,20)
    #ax2.xaxis.set_ticks(np.arange(0,30,5))
    #ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax2.set_xlabel(r'icg iteration $j$',fontsize=30, labelpad=10)
    #ax2.set_ylabel(r'parareal iterations $k$',fontsize=30, labelpad=10)


    # uncomment if using || Ep ||_A^{-1}
    #ax1.plot(np.arange(0,it),q_diff,'brown',ls=(0,(5,1)))
    #ax1.annotate(r'$|\, J(\mathbf{x}_j) - J(\mathbf{x}_\ast)|$',xy=(15,9e-8),fontsize=22,c='brown')
    #ax1.plot(np.arange(0,it),res,'r',ls=(0,(3,2,1,2)))
    #ax1.annotate(r'$\Vert \mathbf{r}_j \Vert_{\mathbf{A}^{-1}}$',xy=(21,1.8e-3),fontsize=22,c='r')
    #ax1.plot(np.arange(0,it),e_bd,'m',ls=(0,(3,1,1,1,1,1)))
    #ax1.annotate(r'$\omega_j$',xy=(23,0.025),fontsize=22,c='m')
    #ax1.plot(np.arange(0,it),e_bd2,c='orange',ls=(30,(3,2,1,2,1,2)))
    #ax1.annotate(r'$\xi_j$',xy=(22,8e-6),fontsize=22,c='orange')    
    #ax1.plot(np.arange(0,it),gap1,'g',ls=(0,(1,3)))
    #ax1.annotate(r'$\Vert \mathbf{r}(\mathbf{x}_j)-\mathbf{r}_j \Vert_{\mathbf{A}^{-1}}$',xy=(19,1e-4),fontsize=22,c='g')
    #ax1.xaxis.set_ticks(np.arange(0,26,5))
    #ax2.plot(cg[8],'o--')
    #ax2.set_ylim(1,9)
    #ax2.xaxis.set_ticks(np.arange(0,30,5))
    #ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax2.set_xlabel(r'icg iteration $j$',fontsize=30, labelpad=10)
    #ax2.set_ylabel(r'parareal iterations $k$',fontsize=30, labelpad=10)


    # uncomment if using approximate || Ep ||_A^{-1} and all other approximations
    ax1.plot(np.arange(0,it-2),q_diff[:-2],'brown',ls=(0,(5,1)))
    ax1.annotate(r'$|\, J(\mathbf{x}_j) - J(\mathbf{x}_\ast)|$',xy=(15,9e-8),fontsize=22,c='brown')
    ax1.plot(np.arange(0,it-2),res[:-2],'r',ls=(0,(3,2,1,2)))
    ax1.annotate(r'$\Vert \mathbf{r}_j \Vert_{\mathbf{A}^{-1}}$',xy=(21,1.8e-3),fontsize=22,c='r')
    ax1.plot(np.arange(0,it-2),e_bd[:-2],'m',ls=(0,(3,1,1,1,1,1)))
    ax1.annotate(r'$\omega_j$',xy=(23,0.025),fontsize=22,c='m')
    ax1.plot(np.arange(0,it-2),e_bd2[:-2],c='orange',ls=(30,(3,2,1,2,1,2)))
    ax1.annotate(r'$\xi_j$',xy=(22,8e-6),fontsize=22,c='orange')    
    ax1.plot(np.arange(0,it-2),gap1[:-2],'g',ls=(0,(1,3)))
    ax1.annotate(r'$\Vert \mathbf{r}(\mathbf{x}_j)-\mathbf{r}_j \Vert_{\mathbf{A}^{-1}}$',xy=(19,1e-4),fontsize=22,c='g')
    ax1.xaxis.set_ticks(np.arange(0,26,5))
    ax2.plot(cg[8][:-2],'o--')
    ax2.set_ylim(4,9)
    ax2.xaxis.set_ticks(np.arange(0,30,5))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel(r'icg iteration $j$',fontsize=30, labelpad=10)
    ax2.set_ylabel(r'parareal iterations $k$',fontsize=30, labelpad=10)
    


    fig.tight_layout()
    plt.savefig('figure.pdf',format = 'pdf',dpi=1000)



#------------------- other functions for plots ------------------------------#
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
def initial_states(nx,lambda0,cgsol,icgsol):
    f1, (a1,a2) = plt.subplots(1,2,figsize=(18,8),sharey=True)

    a1.margins(x=0.04)
    a1.tick_params(bottom=True, top=True, left=True, right=True)
    a1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    a1.hlines(1.,0,121,'w')
    a1.set_xticks(np.arange(0,121,30))
    a1.plot(lambda0[:nx],'r',ls=(0,(1,1,1,1,1)),label=r'Reference')
    a1.plot(cgsol[0][:nx],'b-.',label='CG')
    a1.plot(icgsol[0][:nx],'m--',label='Inexact CG')
    a1.set_ylabel('values',fontsize=30)
    a1.set_xlabel('$x$',fontsize=30)
    a1.legend(fontsize=22)

    a2.tick_params(bottom=True, top=True, left=True, right=True)
    a2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    a2.margins(x=0.025)
    a2.hlines(1.,0,121,'w')
    a2.set_xticks(np.arange(0,121,30))
    a2.plot(lambda0[nx:],'r',ls=(0,(1,1,1,1,1,)),label=r'Reference')
    a2.plot(cgsol[0][nx:],'b-.',label='CG')
    a2.plot(icgsol[0][nx:],'m--',label='Inexact CG')
    a2.set_xlabel('$x$',fontsize=30)
    a2.legend(fontsize=22)
    f1.tight_layout()
    plt.savefig('initial_states.pdf',format = 'pdf',dpi=1000)



def contourplots(F,N,Dt,L,nx,icgsol,refsol):
    
    t = np.linspace(0,N*Dt,N+1)
    t = np.array(t,dtype=float)
    
    x = np.linspace(0,L,int(L))
    x = np.array(x,dtype=float)
    
    X,T = np.meshgrid(x,t)
    
    # for free surface
    Z1 = icgsol[:,:nx]
    Z1 = np.array(Z1,dtype=float)
    
    Z2 = refsol[:,:nx]
    Z2 = np.array(Z2,dtype=float)

    f1,a1 = plt.subplots(figsize=(14,10),layout='tight')
    plt.contourf(X,T,Z1,cmap='cividis')
    a1.set_xlabel('x')
    a1.set_ylabel('t')
    a1.set_title(r'Inexact CG solution: $\eta$')
    plt.colorbar()
    plt.savefig('contour_icg_eta.pdf',format = 'pdf',dpi=1000)

    f2,a2 = plt.subplots(figsize=(14,10),layout='tight')
    a2.set_xlabel('x')
    a2.set_ylabel('t')
    a2.set_title(r'Exact solution: $\eta$')
    plt.contourf(X,T,Z2,cmap='cividis')
    plt.colorbar()
    plt.savefig('contour_exact_eta.pdf',format = 'pdf',dpi=1000)

    # for veloctiy
    x = np.linspace(0,L,int(L-1))
    x = np.array(x,dtype=float)

    X,T = np.meshgrid(x,t)

    Z3 = icgsol[:,nx:]
    Z3 = np.array(Z3,dtype=float)
    
    Z4 = refsol[:,nx:]
    Z4 = np.array(Z4,dtype=float)

    f3,a3 = plt.subplots(figsize=(14,10),layout='tight')
    plt.contourf(X,T,Z3,cmap='viridis')
    a1.set_xlabel('x')
    a1.set_ylabel('t')
    a1.set_title(r'Inexact CG solution: $u$')
    plt.colorbar()
    plt.savefig('contour_icg_u.pdf',format = 'pdf',dpi=1000)

    f4,a4 = plt.subplots(figsize=(14,10),layout='tight')
    a2.set_xlabel('x')
    a2.set_ylabel('t')
    a2.set_title(r'Exact solution: $u$')
    plt.contourf(X,T,Z4,cmap='viridis')
    plt.colorbar()
    plt.savefig('contour_exact_u.pdf',format = 'pdf',dpi=1000)
