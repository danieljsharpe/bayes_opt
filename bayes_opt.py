'''
Pure Python implementation of Bayesian Optimisation (BO) for a continuous objective function.
Features a variety of acquisition functions to deal with noise etc.
The objective function is flexible, e.g. it can be the output of a simulation
Daniel J. Sharpe
Oct 2018
'''

import matplotlib.pyplot as plt
import numpy as np

''' Class containing a set of acquisition functions, one of which will be required by Bayes_Opt '''
class Acquisition_Funcs(object):

    def __init__(self, costfunc, domain, af_choice):
        self.costfunc = costfunc # the objective (cost) function
        self.domain = domain     # domain (list of (min,max) of each coord) of objective function
        # set acquisition function
        if af_choice == 0: self.acq_func = self.expec_impv
        elif af_choice == 1: self.acq_func = self.knowledge_gradient

    def acq_main_loop(self, x):
        self.acq_func(x)

    def expec_impv(self, x, niter):
        print "Called expected improvement!"
        fval_best = np.max(self.obs_fvals[:niter])
        print "The current best value of fval is:", fval_best
        # find x that maximises expected improvement ei
        ei = 1.

    def knowledge_gradient(self, x):
        pass

    ''' Return vector of n random numbers given a domain '''
    @staticmethod
    def uniform_randno(domain):
        random_coords = []
        for i in range(len(domain)):
            random_coords.append(np.random.uniform(domain[i][0],domain[i][1]))
        return np.array(random_coords)

''' Class containing functions required for Gaussian process regression (GPR) '''
class GPR(object):

    def __init__(self, gpr_arg1, gpr_arg2):
        # set mean function (of the prior) and kernel function defining the Gaussian process
        if gpr_arg1 == 1:
            self.mean_func = self.const_mean_func
            self.mu_0 = 0.
        if gpr_arg2 == 1: self.kernel = self.gaussian_kernel
        self.alpha = [0.5] + [100.]*(self.ndim)

    def gpr_main_loop(self):
        print "In main loop of GPR!"

    ''' Function to place a Gaussian process prior on the objective function '''
    def place_prior(self):
        self.xspace = np.zeros((self.ndim,self.n_grid),dtype=float)
        for i in range(self.ndim):
            self.xspace[i,:] = [self.domain[i,0] + ((float(j)/float(self.n_grid)) \
                                *self.domain[i,1]) for j in range(self.n_grid)]
        for i in range(self.n_grid):
            self.means[i] = self.mean_func(self.xspace[:,i],self.mu_0)
            for j in range(self.n_grid):
                self.covar[i,j] = self.kernel(self.xspace[:,i],self.xspace[:,j],self.alpha)
        self.covar += 1.E-05*np.eye(self.n_grid)
        print "mean vector of prior:\n", self.means
        print "covar mtx of prior:\n", self.covar

    ''' Function to update the posterior distribution of a new observation, given a set of previous
        observations, according to Bayes' rule  '''
    def update_posterior(self, x, n_iter):
        covar_vec_new = np.zeros(n_iter,dtype=float) # covariance vec for the proposed observation at x
        for i in range(n_iter):
            covar_vec_new[i] = self.kernel(x,self.obs_coords[i],self.alpha)
        print "covar:\n", self.covar[:n_iter,:n_iter]
        mu_n = np.dot(np.dot(covar_vec_new,np.linalg.inv(self.covar[:n_iter,:n_iter])),
                      (self.obs_fvals[:n_iter]-self.means[:n_iter])) + self.means # should be "mu_0" ?
        var_n = self.kernel(x,x,self.alpha) - np.dot(np.dot(covar_vec_new, \
                                                 np.linalg.inv(self.covar[:n_iter,:n_iter])),
                                          covar_vec_new.transpose())
        print "mu_n, var_n:", mu_n, var_n
        return mu_n, var_n

    ''' Function to update the posterior distribution via Cholesky decomposition of the covariance matrix '''
    def cholesky_update_posterior(self):
        K = np.zeros((self.n_init_obs,self.n_init_obs),dtype=float)
        for i in range(self.n_init_obs):
            for j in range(self.n_init_obs):
                K[i,j] = self.kernel(self.obs_coords[i],self.obs_coords[j],self.alpha)
        L = np.linalg.cholesky(K + 1.E-05*np.eye(self.n_init_obs))
        K_s = np.zeros((self.n_init_obs,self.n_grid),dtype=float)
        for i in range(self.n_init_obs):
            for j in range(self.n_grid):
                K_s[i,j] = self.kernel(self.obs_coords[i],self.xspace[:,j],self.alpha)
        Lk = np.linalg.solve(L,K_s)
        self.means = np.dot(Lk.T, np.linalg.solve(L,self.obs_fvals))
        self.covar -= np.dot(Lk.T,Lk)
        print "covariance matrix is now:\n", self.covar

    ''' Function to draw nsamples random functions from the Gaussian process: given n known points x, the
        mean vector mu, and the covariance matrix covar from the kernel function '''
    def sample_gp(self, nsamples):
        return np.array(np.random.multivariate_normal(self.means,self.covar,size=nsamples))

    ''' Function to sample from the Gaussian process via the Cholesky decomposition of the covariance
        matrix. Note that the resulting vector has dimensions of the observed dataset (small) (?) '''
    def cholesky_sample_gp(self, nsamples):
        A = np.linalg.cholesky(self.covar)
        # f = np.zeros((nsamples,n_iter),dtype=float)
        # for i in range(nsamples):
            # f[i,:] = np.dot(A, np.random.normal(size=n_iter))
        f = self.means + np.dot(np.random.normal(size=(nsamples,self.n_grid)),A)
        return f

    ''' Kernel functions (must be positive semi-definite) for evalutating elements of the
        covariance matrix '''
    # xp is redundant (?)
    @staticmethod
    def gaussian_kernel(x, xp, alpha):
        '''
        print "x:\n", x, "\nxp:\n", xp, "\nx[0]:\n", x[0], "\nshape(x):", np.shape(x)
        ndim = len(x[0])
        n = len(x)
        x_arr = np.repeat(x,n,axis=1)
        print "x_arr:\n", x_arr
        return np.array(alpha[0]*np.exp(-0.5*np.dot(x-xp,alpha[1:])))
        '''
        s = alpha[0]*np.exp(-0.5*np.sum([alpha[i+1]*(x[i]-xp[i])**2 for i in range(len(x))]))
        return s 
        # return np.array(alpha[0]*np.exp(-0.5*0.2*np.dot(x,xp.transpose())))

    @staticmethod
    def gauss_sqdist_kernel(x, xp, alpha):
        return 1.

    @staticmethod
    def matern1_kernel(x, xp, alpha, nu):
        s = 1.
        return s

    ''' Mean functions for evaluating the mean vector mu_0 for the prior distribution '''
    @staticmethod
    def const_mean_func(x, mu):
        return np.array([mu]*len(x),ndmin=1)

''' Main class for Bayesian Optimisation. Uses the methods of Gaussian process regresssion and
acquisition function classes '''
class Bayes_Opt(GPR, Acquisition_Funcs):

    def __init__(self, gpr_args, acq_func_args, bo_args):
        np.random.seed(17)
        self.ndim = len(domain)      # dimensionality of objective function
        self.n_init_obs = bo_args[0] # no. of initial observations to be made on objective func
        self.n_obs = bo_args[1]      # no. of subsequent observations to be made on objective func
        self.n_grid = bo_args[2]     # no. of grid points on which to define the GP (defines dims
                                     # of mean vector and covariance matrix)
        self.obs_coords = np.zeros((self.n_init_obs+self.n_obs,self.ndim),dtype=float)
        self.obs_fvals = np.zeros(self.n_init_obs+self.n_obs,dtype=float)
        self.means = np.zeros(self.n_grid,dtype=float) # vector of mean values
        self.covar = np.zeros([self.n_grid]*2,dtype=float) # covariance matrix
        Acquisition_Funcs.__init__(self, *acq_func_args)
        GPR.__init__(self, *gpr_args)
        print "Finished initialising"

    def bo_main_loop(self):
        #self.gpr_main_loop()
        #self.acq_main_loop(x)
        self.place_prior()
        self.init_obs()
        for i in range(self.n_obs):
            # self.cholesky_update_posterior()
            pass

    ''' Make initial observations on the objective function '''
    def init_obs(self):
        for i in range(self.n_init_obs):
            self.obs_coords[i,:] = Acquisition_Funcs.uniform_randno(self.domain)
            self.obs_fvals[i] = self.costfunc(self.obs_coords[i,:])
        # print "Coordinates of observations:\n", self.obs_coords
        # print "Objective function values of observations:\n", self.obs_fvals

if __name__ == "__main__":

    '''
    # test case: Hosaki's function (negative of) (takes 2D list of args)
    costfunc = lambda x: ((1. - (8.*x[0]) + (7.*x[0]**2) - ((7./3.)*x[0]**3) + 
                          (1./4.)*x[0]**4)*(x[1]**2)*np.exp(-x[1]))*(-1.)
    domain = ((0.,5.),(0.,5.))


    bayes_opt1 = Bayes_Opt((1,1), (costfunc,domain,0), (10,10))
    bayes_opt1.bo_main_loop()
    '''

    #'''
    # Simple 1D example with complete enumeration of the acquisition function
    # sum of three Gaussians cost function
    costfunc = lambda x: 0.35*np.exp(-0.5*(1./(0.05**2))*(x[0]-0.22)**2) + \
                         0.40*np.exp(-0.5*(1./(0.14**2))*(x[0]-0.56)**2) + \
                         0.55*np.exp(-0.5*(1./(0.06**2))*(x[0]-0.80)**2)
    domain = np.array([0.,1.])
    domain = np.reshape(domain,(-1,2))
    print domain
    print np.shape(domain), type(domain)
    print domain[0]
    print domain[0,0], domain[0,1]
    n_b4_stop, n_more_obs = 5, 0
    bayes_opt2 = Bayes_Opt((1,1), (costfunc,domain,0), (n_b4_stop,n_more_obs,100))
    bayes_opt2.bo_main_loop()

    print "Made observations at:\n", bayes_opt2.obs_coords, "\nwith function values:\n", \
          bayes_opt2.obs_fvals

    '''
    # after 3 iters, visualise the target & acquisition funcs and posterior distribn
    fval_best = np.max(bayes_opt2.obs_fvals[:n_b4_stop])
    print "Current best:", fval_best
    ei = np.zeros(100,dtype=float)
    '''
    '''
    for i in range(0,100,1):
        x_coord = np.array(i/500.)
        x_coord = np.reshape(x_coord,(-1,1))
        mu_n, var_n = bayes_opt2.update_posterior(x_coord,n_b4_stop)
        delta = max(0,mu_n - fval_best)
        print "mu_n:", mu_n, "var_n", var_n, "delta:", delta 
        # ei[i] = 
    '''
    xspace = np.linspace(0,1,100)

    #'''
    # draw random functions from GP prior and plot
    ## f_prior = bayes_opt2.sample_gp(3)          # sample
    f_prior = bayes_opt2.cholesky_sample_gp(3) # alternative: sample via Cholesky decomposn
    fig_pr = plt.figure()
    ax_pr = plt.axes()
    for i in range(3):
       ax_pr.plot(xspace, f_prior[i,:])
    plt.show()
    #'''

    # plot the 1D costfunction
    fig = plt.figure()
    ax = plt.axes()
    fvals = np.array([costfunc(np.array(x,ndmin=1)) for x in xspace])
    ax.plot(xspace, fvals, "k--", lw=2)
    plt.plot(bayes_opt2.obs_coords,bayes_opt2.obs_fvals,"o",color="red")

    #'''
    # posterior to update mean function and covariance matrix
    bayes_opt2.cholesky_update_posterior()
    print "mu_n (mean vector at current iter) is:\n", bayes_opt2.means
    ax.plot(xspace, bayes_opt2.means, "r--", lw=2)

    # draw random functions from GP posterior
    ## f_post = bayes_opt2.sample_gp(3)            # sample
    f_post = bayes_opt2.cholesky_sample_gp(3) # alternative: sample via Cholesky decomposn
    for i in range(3):
        ax.plot(xspace, f_post[i,:])
    #'''

    plt.show()

    #'''
    '''
    # Can call Acquisition_Funcs and GPR classes on their own
    acq_funcs1 = Acquisition_Funcs(lambda x: x**5, (-5.,+5.), 0)
    print "Calling expec_impv in __main__:"
    acq_funcs1.expec_impv(2.)
    '''
