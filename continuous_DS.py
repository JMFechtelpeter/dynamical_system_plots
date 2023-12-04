import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools as it
from typing import Callable

class PhasePlanePlot:

    def __init__(self, flow_equation: Callable, vectorized: bool=False, 
                 xlim: tuple=None, ylim: tuple=None, ax: mpl.axes.Axes=None, **fig_kwargs):
        '''
        PhasePlanePlot is a class that contains a matplotlib axis and a 2D flow equation.
        Arguments:
        flow_equation:  function defining the flow. It must take x and y as arguments and return dx/dt and dy/dt.
        vectorized:     statement whether the flow_equation is defined in a vectorized fashion, such that it can be given
                        arrays as x and y arguments. This makes calculations faster.
        xlim, ylim:     limits of the plot
        ax:             an existing matplotlib axis. If specified, everything will be plotted there. If not, 
                        PhasePlanePlot will create a new figure.
        fig_kwargs:     Arguments passed to plt.figure
        '''
        self.flow = flow_equation
        self.vectorized = vectorized
        if ax is None:
            self.fig, self.ax = plt.subplots(**fig_kwargs)
        else:
            self.fig = ax.figure
            self.ax = ax
        if xlim is not None and ylim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(xlim)
    
    def trajectory(self, T: float, initial_condition: tuple, t_eval: tuple=None):
        '''
        Integrate the flow from t=0 to t=T. Returns (t, y), where t is the support times and y is the solution at those time points.
        Arguments:
        T:                  Integration limit
        initial_condition:  (x0, y0) tuple
        t_eval:             Force the integrator to evaluate the solution at these time points.
        '''
        x0, y0 = initial_condition
        func = lambda t, xy: self.flow(*xy)
        traj = integrate.solve_ivp(func, (0,T), (x0,y0), t_eval=t_eval)
        return traj.t, traj.y
    
    def plot_flow(self, xpoints: int=20, ypoints: int=20, **plot_kwargs):
        '''
        Plot the flow as arrows.
        Arguments:
        xpoints, ypoints:   How many points will be sampled from the flow to create the plot
        plot_kwargs:        Arguments passed to plt.streamplot
        '''
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xcoord = np.linspace(xlim[0], xlim[-1], xpoints)
        ycoord = np.linspace(ylim[0], ylim[-1], ypoints)
        xmesh, ymesh = np.meshgrid(xcoord, ycoord)
        if self.vectorized:            
            xdotmesh, ydotmesh = self.flow(xmesh, ymesh)
        else:
            xdotmesh = np.zeros_like(xmesh)
            ydotmesh = np.zeros_like(ymesh)
            for i, j in it.product(range(xpoints), range(ypoints)):
                xdotmesh[i,j], ydotmesh[i,j] = self.flow(xmesh[i,j], ymesh[i,j])
        
        self.ax.streamplot(xmesh, ymesh, xdotmesh, ydotmesh, **plot_kwargs)
    
    def plot_curve(self, curve_func: Callable, support_lim: tuple=None, y_support: bool=False, 
                   points: int=100, vectorized: bool=False, **plot_kwargs):
        '''
        Evaluate and plot an arbitrary function.
        Arguments:
        curve_func:     Function definition. Must take 1D input and return 1D output.
        support_lim:    Evaluate the function in these boundaries.
        y_support:      If true, the function is considered to map y onto x. If false, it maps x onto y.
        points:         Number of points to evaluate the function
        vectorized:     statement whether curve_func is defined in a vectorized fashion, such that it can be given
                        arrays as arguments. This makes calculations faster.
        plot_kwargs:    Arguments passed to plt.plot
        '''
        if support_lim is None:
            if y_support:
                support_lim = self.ax.get_ylim()
            else:
                support_lim = self.ax.get_xlim()
        support = np.linspace(*support_lim, points)
        if vectorized:
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
        self.ax.plot(x, y, **plot_kwargs)
    
    def plot(self, x, y, **plot_kwargs):
        '''
        Plot arbitrary values.
        Arguments:
        x, y:           x and y values to plot
        plot_kwargs:    Arguments passed to plt.plot
        '''
        self.ax.plot(x, y, **plot_kwargs)
    
    def plot_axes(self, **plot_kwargs):
        '''
        Plot the x and y axes
        Arguments:
        plot_kwargs:    Arguments passed to plt.plot
        '''
        self.ax.plot(self.ax.get_xlim(), (0,0), **plot_kwargs)
        self.ax.plot((0,0), self.ax.get_ylim(), **plot_kwargs)