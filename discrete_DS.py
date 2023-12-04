from typing import Callable
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class MapPlot:
    def __init__(self, recursive_map: Callable, ax: Axes=None, **fig_kwargs):
        '''
        Base class for map plots.
        Arguments:
        recursive_map:  Map that defines the dynamical system. It must take a 1D argument, and return a 1D output.
        ax:             an existing matplotlib axis. If specified, everything will be plotted there. If not, 
                        PhasePlanePlot will create a new figure.
        fig_kwargs:     Arguments passed to plt.figure
        '''
        self.map = recursive_map
        if ax is None:
            self.fig, self.ax = plt.subplots(**fig_kwargs)
        else:
            self.fig = ax.figure
            self.ax = ax
    
    def plot_curve(self, curve_func, support_lim=None, y_support=False, points=100, vectorized=True, **plot_kwargs):
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

class CobwebPlot(MapPlot):

    def __init__(self, recursive_map: Callable, x0: float, T: int=30,
                 xlim: tuple=None, ax: Axes=None, fig_kwargs: dict={}, plot_kwargs: dict={}):
        '''
        CobwebPlot draws cobweb plots of the a recursive map on an axis.
        Arguments:
        recursive_map:  Map that defines the dynamical system. It must take a 1D argument, and return a 1D output.
        x0:             Initial condition
        T:              Time steps to evaluate the map
        xlim:           limits of the plot
        ax:             an existing matplotlib axis. If specified, everything will be plotted there. If not, 
                        PhasePlanePlot will create a new figure.
        fig_kwargs:     Arguments passed to plt.figure
        plot_kwargs:    Arguments passed to plt.plot
        '''
        super().__init__(recursive_map, ax, **fig_kwargs)
        if xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(xlim)
        traj = self.trajectory(x0, T)
        self.plot_cobweb(traj, **plot_kwargs)

    def trajectory(self, x0: float, T: int):
        '''
        Iterate the recursive map T-1 times from x0. Returns [x0, f(x0), f(f(x0)), ..., f^(T-1)(x0)] 
        '''
        traj = [x0]
        for t in range(1,T):
            traj.append(self.map(traj[-1]))
        return np.array(traj)
    
    def plot_diagonal(self, **plot_kwargs):
        '''
        Plot the diagonal
        '''
        self.ax.plot(self.ax.get_xlim(), self.ax.get_xlim(), **plot_kwargs)
    
    def plot_cobweb(self, traj: ArrayLike, **plot_kwargs):
        '''
        Creates a cobweb plot from a trajectory.
        Arguments:
        traj:           Vector of trajectory
        plot_kwargs:    Arguments passed to plt.plot
        '''
        traj = np.repeat(traj, 2)
        traj_ahead = traj[2:]
        traj = traj[1:-1]
        self.ax.plot(traj, traj_ahead, **plot_kwargs)
        self.ax.plot(traj[0], traj_ahead[0], marker='o')
        self.ax.set_xlabel('x(t)')
        self.ax.set_ylabel('x(t+1)')

class BifurcationPlot(MapPlot):

    def __init__(self, recursive_map: Callable, bifurcation_param: tuple,
                 min_x0: float, max_x0: float, n_trajectories: int=1000, T: int=100,
                 ax: Axes=None, fig_kwargs: dict={}, plot_kwargs: dict={}):
        '''
        BifurcationPlot draws bifurcation plots of the a recursive map on an axis.
        Arguments:
        recursive_map:      Map that defines the dynamical system. It must take arguments (x, b) and return a 1D output. 
                            x is a vector, and b a bifurcation parameter value.                            
        bifurcation_param:  Values of the bifurcation parameter to plot on x axis.
        min_x0, max_x0:     Draw initial conditions from a uniform distribution between these two values
        n_trajectories:     Number of trajectories to draw
        T:                  Iteration time of the map until its value is plotted
        ax:                 an existing matplotlib axis. If specified, everything will be plotted there. If not, 
                            PhasePlanePlot will create a new figure.
        fig_kwargs:         Arguments passed to plt.figure
        plot_kwargs:        Arguments passed to plt.plot
        '''
        super().__init__(recursive_map, ax, **fig_kwargs)
        self.b_values = bifurcation_param
        self.traj_endpoints = np.zeros((len(self.b_values), n_trajectories))
        for i, b in enumerate(self.b_values):
            self.traj_endpoints[i] = self.get_trajectory_endpoints(n_trajectories, T, b, min_x0, max_x0)
        plot_kwargs = {'markersize':0.01, 'marker':'.', 'linestyle': '', **plot_kwargs}
        self.ax.plot(self.b_values, self.traj_endpoints, **plot_kwargs)
        self.ax.set_xlabel('bifurcation param')

    def trajectory(self, x0: float, T: int, b: float):
        '''
        Iterate the recursive map T-1 times from x0. Returns [x0, f(x0), f(f(x0)), ..., f^(T-1)(x0)] 
        Arguments:
        x0:     Initial condition
        T:      Number of iterations
        b:      Value of the bifurcation parameter
        '''
        traj = [x0]
        for t in range(1,T):
            traj.append(self.map(traj[-1], b))
        return np.array(traj)
        
    def get_trajectory_endpoints(self, N: int, T: int, b: float, min_x0: float, max_x0: float):
        '''
        Draw N initial conditions from a uniform distribution between min_x0 and max_x0.
        Iterate the map T times and return the trajectory endpoints. 
        Arguments:        
        N:              Number of initial conditions
        T:              Number of iterations
        b:              Value of the bifurcation parameter
        min_x0, max_x0: Draw initial conditions from a uniform distribution between these two values
        '''
        X0 = np.random.rand(N) * (max_x0 - min_x0) + min_x0
        traj_endpoints = self.trajectory(X0, T, b)[-1]
        return traj_endpoints
    