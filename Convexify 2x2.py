import numpy as np
import ot
import cvxpy as cp

### Data ###
n=3
m=3
mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])
mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])
xs = ot.datasets.make_2D_samples_gauss(n-1, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(m-1, mu_t, cov_t)


### Matrices of distance ###
D1 = ot.dist(xs, xs, metric='Minkowski')
#D1 = np.random.randint(10, size=(n-1,n-1))
#D1 = np.array([[0,1,2],[1,0,3],[2,3,0]])
#D1=np.array([[0,1],[1,0]])

D2 = ot.dist(xs, xs, metric='Minkowski')
#D2 = np.random.randint(10, size=(m-1,m-1))
#D2 = np.array([[0,3,4],[3,0,5],[4,5,0]])
#D2=np.array([[0,2],[2,0]])
D1 /= D1.max()
D2 /= D2.max()

pl.figure()
pl.subplot(121)
pl.imshow(D1)
pl.subplot(122)
pl.imshow(D2)
pl.show()


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
print(Q)



### Construction of H and R ###
Hr = np.reshape(np.repeat(np.eye(n),m), (n, n*m))
Ht = np.tile(np.eye(m),n)
H = np.vstack((Hr, Ht))
Rr = Hr.T@Hr
Rt = Ht.T@Ht
R=H.T@H

print(Q@R-R@Q)#Check that Q and R commute <=> co-diago




### The exact eigenvector of R and its inverse ###
P=np.array([[1,1,1,1,-1,0,-1,0,1],[-1,0,-1,0,-1,-1,0,0,1],[0,-1,0,-1,-1,-1,-1,1,1],
            [-1,-1,0,0,1,1,0,-1,1],[1,0,0,0,1,0,1,-1,1],[0,1,0,0,1,0,0,0,1],
            [0,0,-1,-1,0,1,0,0,1],[0,0,1,0,0,0,1,0,1],[0,0,0,1,0,0,0,1,1]])

P_inv=np.array([[ 1, -2,  1, -2,  4, -2,  1, -2,  1],
 [ 1,  1, -2, -2, -2,  4,  1,  1, -2],
 [ 1, -2,  1,  1, -2,  1, -2,  4, -2],
 [ 1,  1, -2,  1,  1, -2, -2, -2,  4],
 [-2, -2,  1,  1,  1,  4, -2, -2,  1],
 [ 1, -2, -2,  1, -2, -2,  4,  1,  1],
 [-2,  1, -2, -2,  1, -2,  1,  4,  1],
 [-2, -2,  1, -2, -2,  1,  1,  1,  4],
 [ 1,  1,  1,  1,  1,  1,  1,  1,  1]])
P_inv=P_inv/9


print(np.round(P_inv@R@P,3))



### Slight modification of R by eps ###
eps=1e-1

R_val=np.array([eps,eps,eps,eps,3,3,3,3,6]) #[0,0,0,0,3,3,3,3,6] are the eigenvalues of R
R_eps=(P@np.diag(R_val)@P_inv)
"""print(np.round(R_eps,2))
print(np.round(R_eps-R,2))
print(np.round(Q@R_eps-R_eps@Q),10)"""


### Definition of QR_eps = Q+2*lam*R_eps ###
lam=-min(np.linalg.eigvals(-4*Q))/eps # In theory the best lambda to take
lam=1000 # the problem can be non convex is lam is too small -> error do not respect DCP rules
QR_eps=-Q+lam*R_eps
#print(np.linalg.eigvals(QR_eps))

### Weights of points ###
a=np.array([0,80,20])/100 # the first weight is 0 because it is only define for the embedding
b=np.array([0,20,80])/100


### Construct the linear part of Gromov ### 
c=C1**2 @ np.tile(a,(3,1)).T + np.tile(b,(3,1))@ C2**2 # Co
c=np.reshape(c,(1,9))
y_H=np.concatenate((a,b))@H
q=c-2*lam*y_H



### CVX resolution of pq+pQR_epsp ###
p = cp.Variable(9)
objective = cp.Minimize(cp.quad_form(p,QR_eps)+q@p)

A=np.diag(np.array([1,1,1,1,0,0,1,0,0])) #Constraint p to be nul out of the "real support"
constraints = [0 <= p, A@p == 0]

prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.MOSEK,verbose=False)

G1 = p.value
gw_app=np.array([[G1[4],G1[5]], #The real support is here
                 [G1[7],G1[8]]
    ])
print(prob.value)



#############################################################################

### Compute Gromov-Wasserstein plans with pot ###
p=a[1:]
q=b[1:]
gw0, log0 = ot.gromov.gromov_wasserstein(
    D1, D2, p, q, 'square_loss', verbose=True, log=True)


###########################################################################

### Plot the different plans ###
a_tens_b= np.tile(q,(2,1))*np.tile(p,(2,1)).T #a tensor b

#Compare Gromov pot and Gromov convex
pl.figure(1, (10, 5))
pl.subplot(1, 2, 1)
pl.imshow(gw0,vmin=0,vmax=1)
pl.title('Gromov pot')
pl.subplot(1, 2, 2)
pl.imshow(gw_app,vmin=0,vmax=1)
pl.title('Gromov Convex')

#Compare Gromov convex and a tensor b
pl.figure(2, (10, 5))
pl.subplot(1, 2, 1)
pl.imshow(gw_app,vmin=0,vmax=1)
pl.title('Gromov convex')
pl.subplot(1, 2, 2)
pl.imshow(a_tens_b,vmin=0,vmax=1 )
pl.title('a tenseur b')

print(gw0)
print(gw_app)
print(a_tens_b)
