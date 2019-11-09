# implementation of a multivariate normal

import numpy as np

# example: a 2D multivariate normal
# mean vector for the multivariate normal
mean_vec1 = np.array([1.,3.])
# covariance matrix (must be positive semi-definite) for the multivariate normal
rho = 0.1 # correlation parameter
sigma_x, sigma_y = 2.2, 2.8 # std dev in x and y directions
covar_mtx1 = np.array([[sigma_x**2,rho*sigma_x*sigma_y],[rho*sigma_x*sigma_y,sigma_y**2]])

class multiv_norm_func(object):

    def __init__(self,mean_vec,covar_mtx):
        self.mean_vec = mean_vec
        self.covar_mtx = covar_mtx

    # a multivariate normal
    mn_func = lambda self, x: (1./((np.sqrt(((2.*np.pi)**2)*np.linalg.det(self.covar_mtx))))) * \
                               np.exp(-0.5*np.dot(np.dot(np.transpose(x-self.mean_vec),np.linalg.inv(self.covar_mtx)), \
                               x-self.mean_vec))

Mn1 = multiv_norm_func(mean_vec1,covar_mtx1)

print Mn1.mn_func([1.,3.])

# we could have a sum of multivariate Gaussians using the class instantiated with different arguments
