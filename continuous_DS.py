import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import itertools as it

class ContinuousDynamicalSystem2D:

    def __init__(self, xdot, ydot, xlim, ylim, vectorized=False):
        self.xdot = xdot
        self.ydot = ydot
        self.xlim = xlim
        self.ylim = ylim
        self.vectorized = vectorized

    def __call__(self, x, y, t=None, **params):
        return (self.xdot(x, y, t=t, **params), self.ydot(x, y, t=t, **params))
    
    def trajectory(self, T: float, x0: float, y0: float, t_eval: ArrayLike=None, **params):
        func = lambda t, xy: self(*xy, t=t, **params)
        traj = integrate.solve_ivp(func, (0,T), (x0,y0), t_eval=t_eval)
        return traj.t, traj.y
    
    def plot_quiver(self, xlim=None, ylim=None, t=None, xpoints=20, ypoints=20, **params):
        if xlim is None:
            xlim = self.xlim
        if ylim is None:
            ylim = self.ylim
        xcoord = np.linspace(xlim[0], xlim[-1], xpoints)
        ycoord = np.linspace(ylim[0], ylim[-1], ypoints)
        xmesh, ymesh = np.meshgrid(xcoord, ycoord)
        if self.vectorized:            
            xdotmesh, ydotmesh = self(xmesh, ymesh, t=t, **params)
        else:
            xdotmesh = np.zeros_like(xmesh)
            ydotmesh = np.zeros_like(ymesh)
            for i, j in it.product(range(xpoints), range(ypoints)):
                xdotmesh[i,j], ydotmesh[i,j] = self(xmesh[i,j], ymesh[i,j], t=t, **params)
        
        ax = self.get_ax(ax)
        ax.quiver([xmesh, ymesh], xdotmesh, ydotmesh)
        return ax
    
    def plot_curve(self, curve_func, support_lim=None, y_support=False, points=100, ax=None, **plot_kwargs):
        if support_lim is None:
            if y_support:
                support_lim = self.ylim
            else:
                support_lim = self.xlim
        support = np.linspace(*support_lim, points)
        if self.vectorized:
            curve = curve_func(support)
        else:
            curve = np.zeros(points)
            for i in range(points):
                curve[i] = curve_func(support[i])
        if y_support:
            x = curve
            y = support
        else:
            x = support
            y = curve
        ax = self.get_ax(ax)
        ax.plot(x, y, **plot_kwargs)
        return ax
    
    def plot_trajectory(self, x, y, ax=None, **plot_kwargs):
        ax = self.get_ax(ax)
        ax.plot(x, y, **plot_kwargs)
        return ax
    
    def plot_axes(self, ax=None, **plot_kwargs):
        ax = self.get_ax(ax)
        ax.plot(self.xlim, (0,0), **plot_kwargs)
        ax.plot((0,0), self.ylim, **plot_kwargs)
        return ax

        
    def get_ax(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
        return ax