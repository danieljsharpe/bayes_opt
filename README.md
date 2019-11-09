# bayes_opt

TO-DO:

-allow different initial space-filling method for initial observations (besides totally random) (e.g. uniform)
-allow initial observations to be provided as a list
-acquistion functions e.g. EI
-get 1D and 2D examples working
-need to be able to update posterior based on a new set of observations - not just n_init_obs
-MLE for hyperparameter estimation
-KG & ES acquisition funcs
-other kernels e.g. Matern1, Matern2, sqdist...

-expand to include the related method of KRR (?)

BUGS:



OTHER NOTES:

-Sampling from the posterior GP via the Cholesky decomposition gives very "kinked" curves compared to the other method - 
is there a reason for this?
