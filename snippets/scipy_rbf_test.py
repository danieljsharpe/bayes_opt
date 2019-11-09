'''
Example code to demonstrate use of radial basis functions (from scipy.interpolate) to interpolate an
N-dimensional function (here, 2D for illustrative purposes) from an irregular grid of points
'''

from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import numpy as np

# Simple 2D example: sum of three Gaussians objective function
costfunc = lambda x: 0.35*np.exp(-0.5*(1./(0.05**2))*(x[0]-0.22)**2 + \
                                 -0.5*(1./(0.02**2))*(x[1]-0.25)**2) + \
                     0.40*np.exp(-0.5*(1./(0.14**2))*(x[0]-0.56)**2 + \
                                 -0.5*(1./(0.08**2))*(x[1]-0.61)**2) + \
                     0.55*np.exp(-0.5*(1./(0.06**2))*(x[0]-0.80)**2 + \
                                 -0.5*(1./(0.05**2))*(x[1]-0.71)**2)

domain = [[0.,1.],[0.,1.]]

# note for rbf the data does not have to be regularly distributed
np.random.seed(19)
n_test_pts = 100
test_pts = np.array([[np.random.uniform(domain[0][0],domain[0][1]),
                      np.random.uniform(domain[1][0],domain[1][1])]for i in range(n_test_pts)])
test_pts_fvals = np.array([costfunc(test_pt) for test_pt in test_pts])


# instance of Rbf interpolation
rbfi = Rbf(test_pts[:,0],test_pts[:,1],test_pts_fvals,function="gaussian")


# use interpolation to determine approximations for a random set of new values
n_interp_pts = 100
interp_pts = np.array([[np.random.uniform(domain[0][0],domain[0][1]),
                        np.random.uniform(domain[1][0],domain[1][1])]for i in range(n_interp_pts)])
interp_pts_interpvals = rbfi(interp_pts[:,0],interp_pts[:,1])

# calc avg error on these points compared to the actual cost function
interp_pts_fvals = np.array([costfunc(interp_pt) for interp_pt in interp_pts])
err = 0.
for i in range(n_interp_pts):
    err += np.abs(interp_pts_fvals[i]-test_pts_fvals[i])
print "Average error (%):\t", (err/float(n_interp_pts))*100.


# grid points for plotting
grid_pts_x = np.linspace(domain[0][0],domain[0][1],100)
grid_pts_y = np.linspace(domain[1][0],domain[1][1],100)
X,Y = np.meshgrid(grid_pts_x,grid_pts_y)
Z_costfunc = costfunc((X,Y))
Z_interpolated = rbfi(X,Y)

# plot 
fig, _axs = plt.subplots(nrows=1,ncols=2)#,sharex=True,sharey=True)#,figsize=(10,10))
fig.subplots_adjust(hspace=0.3)
axs = _axs.flatten()
cset1 = axs[0].contourf(X,Y,Z_costfunc,50)
axs[0].plot(test_pts[:,0],test_pts[:,1],"o",color="k")
axs[0].set_title("Actual cost function",fontsize=8)
cset2 = axs[1].contourf(X,Y,Z_interpolated,50)
axs[1].set_title("Approximation to cost function by RBFs, given an irregular grid of\n \
                  observation points on the original cost function",fontsize=8)
asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
axs[1].set_aspect(asp)
axs[0].set_aspect(asp)
fig.colorbar(cset1,ax=axs[0],fraction=0.046,pad=0.04)
fig.colorbar(cset2,ax=axs[1],fraction=0.046,pad=0.04)
plt.show()
