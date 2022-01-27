import numpy as np
import ot
import scipy
import sympy as sy
import cvxpy as cp
import matplotlib.pylab as pl
#%matplotlib qt 


### Data ###
n=4 #3 points but 4 dimensional embedding
m=4
mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])
mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])
xs = ot.datasets.make_2D_samples_gauss(n-1, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(m-1, mu_t, cov_t)
#xs=np.array([[3,0],[23,23],[103,103]])
#xt=np.array([[0,0],[20,20],[100,100]])

### Matrices of distance ###
D1 = ot.dist(xs, xs, metric='euclidean')
D2 = ot.dist(xt, xt, metric='euclidean')

k=6
lam_c=np.zeros(k-1)
Q_max=np.zeros(k-1)
Q_min=np.zeros(k-1)
R_max=np.zeros(k-1)
R_min=np.zeros(k-1)


for i in range(1,k):
            
    D1 *= 10**(i-i/2)
    D2 *= 10**(i-i/2)
    
    
    ### Embedding ###
    C1=np.zeros((n,n))
    C1[1:,1:]= D1
    C1[0,1:]=-sum(D1) #first row
    C1[1:,0]=-sum(D1.T) #first column
    C1[0,0] = sum(sum(D1))
    
    C2=np.zeros((m,m))
    C2[1:,1:]=D2
    C2[0,1:]=-sum(D2) #first row
    C2[1:,0]=-sum(D2.T) #first column
    C2[0,0]=sum(sum(D2))
    
    Q=-2*np.kron(C1,C2)
    
    ### Construction of H and R ###
    Hr = np.reshape(np.repeat(np.eye(n),m), (n, n*m))
    Ht = np.tile(np.eye(m),n)
    H = np.vstack((Hr, Ht))
    Rr = Hr.T@Hr
    Rt = Ht.T@Ht
    R=H.T@H
    

    
    
    ### The exact eigenvector of R and its inverse ###
    P=([
     [ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  0, -1, -1,  1, 1],
     [-1,  0,  0, -1,  0,  0, -1,  0,  0, -1, -1, -1,  0, -1,  1, 1],
     [ 0, -1,  0,  0, -1,  0,  0, -1,  0, -1, -1, -1, -1,  0,  1, 1],
     [ 0,  0, -1,  0,  0, -1,  0,  0, -1, -1, -1, -1, -1, -1,  2, 1],
     [-1, -1, -1,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0, -1, 1],
     [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0, -1, 1],
     [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1, -1, 1],
     [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0, 1],
     [ 0,  0,  0, -1, -1, -1,  0,  0,  0,  0,  1,  1,  0,  0, -1, 1],
     [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0, -1, 1],
     [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1, -1, 1],
     [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0, 1],
     [ 0,  0,  0,  0,  0,  0, -1, -1, -1,  0,  0,  1,  0,  0,  0, 1],
     [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0, 1],
     [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0, 1],
     [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1, 1]])
    
    P_inv=np.array([
     [ 1, -3,  1,  1, -3,  9, -3, -3,  1, -3,  1,  1,  1, -3,  1,  1],
     [ 1,  1, -3,  1, -3, -3,  9, -3,  1,  1, -3,  1,  1,  1, -3,  1],
     [ 1,  1,  1, -3, -3, -3, -3,  9,  1,  1,  1, -3,  1,  1,  1, -3],
     [ 1, -3,  1,  1,  1, -3,  1,  1, -3,  9, -3, -3,  1, -3,  1,  1],
     [ 1,  1, -3,  1,  1,  1, -3,  1, -3, -3,  9, -3,  1,  1, -3,  1],
     [ 1,  1,  1, -3,  1,  1,  1, -3, -3, -3, -3,  9,  1,  1,  1, -3],
     [ 1, -3,  1,  1,  1, -3,  1,  1,  1, -3,  1,  1, -3,  9, -3, -3],
     [ 1,  1, -3,  1,  1,  1, -3,  1,  1,  1, -3,  1, -3, -3,  9, -3],
     [ 1,  1,  1, -3,  1,  1,  1, -3,  1,  1,  1, -3, -3, -3, -3,  9],
     [-2, -2, -2,  2,  2,  2,  2,  6, -2, -2, -2,  2, -2, -2, -2,  2],
     [-2, -2, -2,  2, -2, -2, -2,  2,  2,  2,  2,  6, -2, -2, -2,  2],
     [ 2, -2, -2, -2,  2, -2, -2, -2,  2, -2, -2, -2,  6,  2,  2,  2],
     [-2,  2, -2, -2, -2,  2, -2, -2, -2,  2, -2, -2,  2,  6,  2,  2],
     [-2, -2,  2, -2, -2, -2,  2, -2, -2, -2,  2, -2,  2,  2,  6,  2],
     [-2, -2, -2,  2, -2, -2, -2,  2, -2, -2, -2,  2,  2,  2,  2,  6],
     [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
    P_inv=P_inv/16
    

    
    
    
    ### Slight modification of R by eps ###
    eps=1e-1
    R_val=np.array([eps,eps,eps,eps,eps,eps,eps,eps,eps,4,4,4,4,4,4,8]) #eigvalue of R + eps
    R_eps=(P@np.diag(R_val)@P_inv)
    
    
    ### Construction of QR = Q+2*lam*R_eps ### quadratic part
    
    lam=-float(min(np.linalg.eigvals(Q)))/eps # In theory the best lambda to take
    #lam=11860513  # the problem can be non convex is lam is too small -> error do not respect DCP rules
    QR_eps=Q+lam*R_eps
    #print(np.linalg.eigvals(QR_eps))
    
    
    ### Weights of points ###
    a=np.array([0,33,33,33])/100 # the first weight is 0 because it is only define for the embedding
    b=np.array([0,33,33,33])/100
    
    
    ### Construct the linear par of Gromov ###
    c=(C1@C1) @ np.tile(a,(4,1)).T + np.tile(b,(4,1))@ (C2@C2)
    c=np.reshape(c,(1,16))
    y_H=np.concatenate((a,b))@H
    q=c-2*y_H*lam
    

    lam_c[i-1]=lam
    Q_max[i-1]=np.max(np.linalg.eigvals(Q))
    Q_min[i-1]=np.min(np.linalg.eigvals(Q))
    R_max[i-1]=np.min(np.linalg.eigvals(lam*R_eps))
    R_min[i-1]=np.max(np.linalg.eigvals(lam*R_eps))
    
pl.figure(0,figsize=(8,8))
pl.plot(range(1,k), np.log(R_min-Q_min), 'b', label='Difference entre les plus petites vp de Q et R')
pl.plot(range(1,k), np.log(R_max-Q_max), 'g', label='Difference entre les plus grandes vp de Q et R')
pl.plot(range(1,k),np.log(lam_c),'r')
#pl.plot(range(1,k), R_min, 'g', label='R')

#pl.plot(range(1,k), R_max, 'g', label='R')
pl.legend(loc=2)
pl.title('vp')
