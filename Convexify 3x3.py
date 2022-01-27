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
xs=np.array([[3,0],[23,23],[103,103]])
xt=np.array([[0,0],[20,20],[100,100]])

fig = pl.figure(0,figsize=(8,8))
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Source samples')
pl.show()


### Matrices of distance ###
D1 = ot.dist(xs, xs, metric='euclidean')
#D1 = np.random.randint(10, size=(n-1,n-1))
#D1 = np.array([[0,1,2],[1,0,3],[2,3,0]])
#D1=np.array([[0,1,2],[1,0,3],[2,3,0]],dtype=np.float64)

D2 = ot.dist(xt, xt, metric='euclidean')
#D2 = np.random.randint(10, size=(m-1,m-1))
#D2 = np.array([[0,3,4],[3,0,5],[4,5,0]])
#D2=np.array([[0,2,4],[2,0,3],[4,3,0]],dtype=np.float64)


D1 /= D1.max() # We can play with the size of D1 to play o the size  of lam
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

print(Q@R-R@Q)#Check that there a co-diagonalizable



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

print(np.round(P_inv@R@P,3))



### Slight modification of R by eps ###
eps=1
R_val=np.array([eps,eps,eps,eps,eps,eps,eps,eps,eps,4,4,4,4,4,4,8]) #eigvalue of R
R_eps=(P@np.diag(R_val)@P_inv)

print(np.round(R_eps,2))
print(np.round(R_eps-R,2))
print(np.round(Q@R_eps-R_eps@Q,10))


### Construction of QR = Q+2*lam*R_eps ### quadratic part

lam=-float(min(np.linalg.eigvals(Q)))/eps+1 # In theory the best lambda to take
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





### CVX Resolution of pq+p(QR_eps)p ### 
p = cp.Variable(16)
objective = cp.Minimize(cp.quad_form(p,Q+lam*R_eps)+q@p)

A=np.diag(np.array([1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0])) #Constraint p to be nul out of the "real support"
constraints = [0 <= p, A@p == 0]

prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.MOSEK,verbose=False)
G1 = p.value
gw_app=np.array([[G1[5],G1[6],G1[7]], #The real support
                 [G1[9],G1[10],G1[11]],
                 [G1[13],G1[14],G1[15]]
    ])
print(prob.value)







#############################################################################

### Compute Gromov-Wasserstein plans and distance ###
p=a[1:]
q=b[1:]

gw0, log0 = ot.gromov.gromov_wasserstein(
    D1, D2, p, q, 'square_loss', verbose=False, log=True)


print('Gromov-Wasserstein distances: ' + str(log0['gw_dist']))

##########################################################################

### Plot the Plans ###

a_tens_b= np.tile(q,(3,1))*np.tile(p,(3,1)).T #a tensor b

#Compare Gromov pot and Gromov convex
pl.figure(1, figsize=(10, 5))
pl.subplot(1, 2, 1)
pl.imshow(gw0,vmin=0,vmax=1)
pl.title('Gromov pot')
pl.subplot(1, 2, 2)
pl.imshow(gw_app,vmin=0,vmax=1)
pl.title('Gromov convex')

# Compare Gromov convex and a tensor b
pl.figure(2, figsize=(10,5))
pl.subplot(1, 2, 1)
pl.imshow(gw_app,vmin=0,vmax=1)
pl.title('Gromov convex')
pl.subplot(1, 2, 2)
pl.imshow(a_tens_b,vmin=0,vmax=1)
pl.title('a tenseur b')

print(gw0)
print(gw_app)
print(a_tens_b)

#print("vp Q: "+str(np.linalg.eigvals(Q)))

#print("vp QR_eps: " + str(np.linalg.eigvals(QR_eps)))



print(
np.max(np.linalg.eigvals(Q)),
np.min(np.linalg.eigvals(Q)),
np.min(np.linalg.eigvals(lam*R_eps)),
np.max(np.linalg.eigvals(lam*R_eps)))