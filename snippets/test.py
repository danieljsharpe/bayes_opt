import numpy as np
import matplotlib.pyplot as pl

np.random.seed(19)

# Test data
n = 100
Xtest = np.linspace(0, 1, n).reshape(-1,1)

# define the cost function
costfunc = lambda x: 0.35*np.exp(-0.5*(1./(0.05**2))*(x-0.22)**2) + \
                     0.40*np.exp(-0.5*(1./(0.14**2))*(x-0.56)**2) + \
                     0.55*np.exp(-0.5*(1./(0.06**2))*(x-0.80)**2)


# Define the kernel function
def kernel(a, b, param1, param2):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return param1*np.exp(-.5 * (1/param2) * sqdist)

param1 = 0.5
param2 = 0.01
K_ss = kernel(Xtest, Xtest, param1, param2)

# Get cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1.e-8*np.eye(n))
# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L, np.random.normal(size=(n,3)))

# Now let's plot the 3 sampled functions.
pl.plot(Xtest, f_prior)
pl.axis([0, 1, 0, 1])
pl.title('Three samples from the GP prior')
pl.show()

# Noiseless training data
Xtrain = np.array([0.15, 0.3, 0.45, 0.6, 0.75]).reshape(5,1)
ytrain = np.array([costfunc(Xtrain[i]) for i in range(len(Xtrain))]) # the objective function is actually the sin function!

# Apply the kernel function to our training points
K = kernel(Xtrain, Xtrain, param1, param2)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_s = kernel(Xtrain, Xtest, param1, param2)
Lk = np.linalg.solve(L, K_s)
print "Lk:\n", Lk
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)
# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
print "Taking dot:\n", np.dot(Lk.T,Lk)
print "Covariance matrix is now:\n", L
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))

# calculate the cost function
Ytest = np.array([costfunc(Xtest[i]) for i in range(len(Xtest))])

pl.plot(Xtrain, ytrain, 'bs', ms=8)
pl.plot(Xtest, Ytest, 'k--', lw=3)
pl.plot(Xtest, f_post)
pl.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.axis([0, 1, 0, 1])
pl.title('Three samples from the GP posterior')
pl.show()
