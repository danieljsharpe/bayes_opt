'''
Pure Python implementation of Bayesian Optimisation (BO) for a continuous objective function.
Features a variety of acquisition functions to deal with noise etc.
Daniel J. Sharpe
Oct 2018
'''

import numpy as np

''' Class containing a set of acquisition functions, one of which will be required by Bayes_Opt '''
class Acquisition_Funcs(object):

    def __init__(self, costfunc, domain, af_choice):
        print "Initialising Acquisition_Funcs"
        self.costfunc = costfunc # the objective (cost) function
        self.domain = domain     # domain (list of (min,max) of each coord) of objective function
        self.ndim = len(domain)  # dimensionality of objective function
        ### print "costfunc(2.)", costfunc(2)
        ### print "self.costfunc(2.)", self.costfunc(2)
        # set acquisition function
        if af_choice == 0: self.acq_func = self.expec_impv
        elif af_choice == 1: self.acq_func = self.knowledge_gradient

    def acq_main_loop(self, x):
        self.acq_func(x)

    def expec_impv(self, x, niter):
        print "Called expected improvement!"
        # print "costfunc(x)", self.costfunc(x)
        fval_best = np.max(self.obs_fvals[:niter])
        print "The current best value of fval is:", fval_best

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
        if gpr_arg1 == 1:
            self.mean_func = self.const_mean_func
            self.mu = 1.
        elif gpr_arg1 == 2:
            self.mean_func = self.fval_mean_func
            self.mu = 2. # this needs to be the function value at x
        if gpr_arg2 == 1: self.kernel = self.gaussian_kernel
        self.alpha = [0.2]*(self.ndim+1)
        print "Initialising GPR"

    def gpr_main_loop(self):
        print "In main loop of GPR!"

    ''' Function for calculating the prior distribution '''
    def calc_prior(self):
        for i in range(self.n_init_obs):
            self.means[i] = self.mean_func(self.obs_coords[i],self.mu)
            for j in range(self.n_init_obs):
                self.covar[i,j] = self.kernel(self.obs_coords[i],self.obs_coords[j],self.alpha)
        print "Mean vector after calculation of prior:\n", self.means
        print "Covariance matrix after calculation of prior:\n", self.covar

    ''' Function to update the posterior distribution of a new observation, given a set of previous
        observations, according to Bayes' rule  '''
    def calc_posterior(self, x, n_iter):
        covar_vec_new = np.zeros(n_iter,dtype=float) # covariance vec for the proposed observation at x
        for i in range(n_iter):
            covar_vec_new[i] = self.kernel(x,self.obs_coords[i],self.alpha)
        mu_n = np.dot(np.dot(covar_vec_new,np.linalg.inv(self.covar[:n_iter,:n_iter])),
                      (self.obs_fvals[:n_iter]-self.means[:n_iter])) + 1.0 # should be "mu_0" ?
        var_n = self.kernel(x,x,self.alpha) - np.dot(np.dot(covar_vec_new, \
                                                 np.linalg.inv(self.covar[:n_iter,:n_iter])),
                                          covar_vec_new.transpose())
        print "mu_n, var_n:", mu_n, var_n

    ''' Function to evaluate the prior or posterior '''
    def eval_prob_distrib(self):
        pass

    @staticmethod
    def gaussian_kernel(x, xp, alpha):
        s = alpha[0]*np.exp(-np.sum([alpha[i+1]*(x[i]-xp[i])**2 for i in range(len(x))]))
        return s

    @staticmethod
    def const_mean_func(x, mu):
        return mu

    @staticmethod
    def fval_mean_func(x, fval):
        return fval

''' Main class for Bayesian Optimisation. Uses the methods of Gaussian process regresssion and
acquisition function classes '''
class Bayes_Opt(GPR, Acquisition_Funcs):

    def __init__(self, gpr_args, acq_func_args, bo_args):
        self.ndim = 2
        self.n_init_obs = bo_args[0] # no. of initial observations to be made on objective func
        self.n_obs = bo_args[1]      # no. of subsequent observations to be made on objective func
        self.obs_coords = np.zeros((self.n_init_obs+self.n_obs,self.ndim),dtype=float)
        self.obs_fvals = np.zeros(self.n_init_obs+self.n_obs,dtype=float)
        self.means = np.zeros(self.n_init_obs+self.n_obs,dtype=float) # vector of mean values
        self.covar = np.zeros([self.n_init_obs+self.n_obs]*2,dtype=float) # covariance matrix
        Acquisition_Funcs.__init__(self, *acq_func_args)
        GPR.__init__(self, *gpr_args)
        print "Finished initialising"

    def bo_main_loop(self):
        x = (2.,2.)
        self.init_obs()
        self.calc_prior()
        for i in range(self.n_obs):
            self.acq_func((2.,2.),self.n_init_obs+i)

            self.calc_posterior((2.,2.),10)
        '''
        self.gpr_main_loop()
        self.acq_main_loop(x)
        '''

    ''' Make initial observations on the objective function '''
    def init_obs(self):
        for i in range(self.n_init_obs):
            self.obs_coords[i,:] = Acquisition_Funcs.uniform_randno(self.domain)
            self.obs_fvals[i] = self.costfunc(self.obs_coords[i,:])
        print "Coordinates of observations:\n", self.obs_coords
        print "Objective function values of observations:\n", self.obs_fvals

if __name__ == "__main__":

    # test case: Hosaki's function (negative of) (takes 2D list of args)
    costfunc = lambda x: ((1. - (8.*x[0]) + (7.*x[0]**2) - ((7./3.)*x[0]**3) + 
                          (1./4.)*x[0]**4)*(x[1]**2)*np.exp(-x[1]))*(-1.)
    domain = ((0.,5.),(0.,5.))


    bayes_opt1 = Bayes_Opt((1,1), (costfunc,domain,0), (10,10))
    bayes_opt1.bo_main_loop()

    '''
    # Can call Acquisition_Funcs and GPR classes on their own
    acq_funcs1 = Acquisition_Funcs(lambda x: x**5, (-5.,+5.), 0)
    print "Calling expec_impv in __main__:"
    acq_funcs1.expec_impv(2.)
    '''
