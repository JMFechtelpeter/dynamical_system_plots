from typing import Callable
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class Plot:
    def __init__(self, recursive_map: Callable, ax: Axes=None, **fig_kwargs):
        self.map = recursive_map
        if ax is None:
            self.fig, self.ax = plt.subplots(**fig_kwargs)
        else:
            self.fig = ax.figure
            self.ax = ax
    
    def plot_curve(self, curve_func, support_lim=None, y_support=False, points=100, vectorized=True, **plot_kwargs):
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

class CobwebPlot(Plot):

    def __init__(self, recursive_map: Callable, x0: float, T: int=30, map_params: dict={},
                 xlim: tuple=None, ax: Axes=None, fig_kwargs: dict={}, plot_kwargs: dict={}):
        super().__init__(recursive_map, ax, **fig_kwargs)
        if xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(xlim)
        traj = self.trajectory(x0, T, **map_params)
        self.plot_cobweb(traj, **plot_kwargs)

    def trajectory(self, x0: float, T: int, **params):
        traj = [x0]
        for t in range(1,T):
            traj.append(self.map(traj[-1], t=t, **params))
        return np.array(traj)
    
    def plot_diagonal(self, **plot_kwargs):
        self.ax.plot(self.ax.get_xlim(), self.ax.get_xlim(), **plot_kwargs)
    
    def plot_cobweb(self, traj: ArrayLike, **plot_kwargs):
        traj = np.repeat(traj, 2)
        traj_ahead = traj[2:]
        traj = traj[1:-1]
        self.ax.plot(traj, traj_ahead, **plot_kwargs)
        self.ax.plot(traj[0], traj_ahead[0], marker='o')
        self.ax.set_xlabel('x(t)')
        self.ax.set_ylabel('x(t+1)')

class BifurcationPlot(Plot):

    def __init__(self, recursive_map: Callable, bifurcation_param: dict,
                 min_x0: float, max_x0: float, 
                 n_trajectories: int=1000, T: int=100, map_params: dict={},
                 ax: Axes=None, fig_kwargs: dict={}, plot_kwargs: dict={'marker':'.', 'c':'k', 'markersize':0.01}):
        
        super().__init__(recursive_map, ax, **fig_kwargs)
        b, self.b_values = list(bifurcation_param.items())[0]
        self.traj_endpoints = np.zeros((len(self.b_values), n_trajectories))
        for i, v in enumerate(self.b_values):
            map_params[b] = v
            self.traj_endpoints[i] = self.get_trajectory_endpoints(n_trajectories, T, min_x0, max_x0, **map_params)
        plot_kwargs['linestyle'] = ''
        self.ax.plot(self.b_values, self.traj_endpoints, **plot_kwargs)
        self.ax.set_xlabel(b)

    def trajectory(self, x0: float, T: int,  **params):
        traj = [x0]
        for t in range(1,T):
            traj.append(self.map(traj[-1], t=t, **params))
        return np.array(traj)
        
    def get_trajectory_endpoints(self, N: int, T: int, min_x0: float, max_x0: float, **params):
        X0 = np.random.rand(N) * (max_x0 - min_x0) + min_x0
        traj_endpoints = self.trajectory(X0, T, **params)[-1]
        return traj_endpoints
    
    def find_bifurcations(self, tol):
        traj_endpoint_std = self.traj_endpoints.std(axis=1)
        std_diff = np.diff(traj_endpoint_std, n=10)
        return self.b_values[:len(std_diff)][std_diff > tol]