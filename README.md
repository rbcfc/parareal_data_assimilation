The python file contains the code for implementing Parareal for the 1D linearised shallow water equations as the data assimilation (incremental 4D-Var inner loop) model. The cost function minimisation makes 
use of the inexact conjugate gradient algorithm as described in [1]. The system solved is 

[1] Gratton, S., Simon, E., Titley‐Peloquin, D., & Toint, P. L. (2021). Minimizing convex quadratics with variable precision conjugate gradients. Numerical Linear Algebra with Applications, 28(1), e2337.
