
import matplotlib.pyplot as plt
import numpy as np
from utils_energy import get_energy_solar, get_energy_planet # get_center_of_mass
from pprint import pprint

# bokeh
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.io import output_notebook
from bokeh.palettes import Category20_10 as palette
import itertools
from bokeh.models import Arrow, OpenHead, NormalHead
class Plot:
    # here we are plotting for simulations that have the exact same number of time steps.
    # for the exact same bodies (same mass per body, in the same order)
    """
    Class that provides energy computing and plotting utilities. Functions to compute the energies are defined in the module *utils_energy.py*.
    This class takes into input an array or list of positions of format {number of schemes x number of time steps x number of bodies x 4}.
    *4* for : x, y, vx, vy.
    Note :
        * It is possible to have various lenght in {number of time steps}.
        * It is possible to have various masses (as a function of the scheme used).
    """
    def __init__(self, positions, masses, scheme_names, dt_in_days, theta, fn_evaluations, names = None):
        """ Initialize a plotter by also computing the evolution of energy per body and per schemes and the total energy of the system.
        Various plotting function can be used afterwards to plot : evolution of the system, energy evolutions of the system (per scheme, per body, total energy).

        Args:
            positions (array or list): should be of lenght {number of schemes x number of time steps x number of bodies x 4}.
            masses (list): should be of lenght {number of bodies} or {number of schemes x number of bodies}. If 2D, the i-th scheme received the i-th masses list.
            scheme_names (list of string): should be of lenght {number of schemes}.
            dt_in_days (int or list): if a list or array, the i-th scheme received the i-th dt-in-days.
            names (list of strings, optional): Names of the bodies. Defaults to None.
        """
        if(not self.isnotebook()):
            print('This class should be used inside a notebook.')
            return

        self.constant_masses = True if(len(np.array(masses).shape)==1) else False
        self.masses = np.array([masses]*len(scheme_names)) if self.constant_masses else np.array(masses)

        self.constant_dt = True if(type(dt_in_days)==type(1) or type(dt_in_days)==type(1.0)) else False
        self.dt_in_days = np.array([dt_in_days]*len(scheme_names)) if self.constant_dt else np.array(dt_in_days)

        # size = [schemes, nb_steps, nb_planets,4], nb_steps may vary
        self.positions = [np.array(position) for position in positions]
        s = self.positions[0].shape
        self.number_of_bodies = s[1] # makes no sense if it's not the same of everyone
        # something with all(True) etc.
        self.number_of_steps = np.array([pos.shape[0] for pos in self.positions])

        self.schemes = scheme_names
        self.names = names
        self.fn_evaluations = fn_evaluations
        
        self.theta_list = theta if type(theta) == list else np.array([theta]*len(scheme_names)) # only useful for plotting names

        # init : we compute various data such as the energy etc.
        self.energy_per_body = self._compute_energy() # contains energy ordered by schemes, then body then time step
        self.energy_sum = self._compute_sum_energy() # contains energy ordered by schemes then time step.

        output_notebook()

    def __repr__(self):
        string = "Plotter :\n\tSchemes : {}\n\tNumber of simulations : {} \n\tNumber of bodies : {} \n\tNumber of time steps : {} \n\tTime step : {} day".format(', '.join(self.schemes), len(self.schemes), self.number_of_bodies, ', '.join([str(o) for o in self.number_of_steps]), ', '.join([str(dt) for dt in self.dt_in_days]))
        string_ = '\n\tBody names : {}'.format(', '.join(self.names)) if self.names!=None else ''
        return string + string_

    def plot_positions(self, ncols = None, print_legend = False, print_arrow = True, idx_ref = None):
        grid = []
        for k, pos_ in enumerate(self.positions):
            if(k != idx_ref):
                pos = np.moveaxis(pos_, 0, -1)  # [nb_planets,4,nb_steps]

                fig = figure(plot_width=980, plot_height=500, match_aspect=True, \
                    title="{} ; dt = {} days ; theta = {}".format(self.schemes[k], self.dt_in_days[k], self.theta_list[k]), \
                        x_axis_label='x', y_axis_label='y',)
                colors = itertools.cycle(palette) 
                for body_idx in range(self.number_of_bodies):
                    c = next(colors)
                    if(self.names!=None):
                        fig.line(x=pos[body_idx][0],y=pos[body_idx][1], color = c, legend_label = self.names[body_idx])
                    else :
                        fig.line(x=pos[body_idx][0],y=pos[body_idx][1], color = c)
                    l = len(pos[body_idx][0])
                    if(print_arrow):
                        fig.add_layout(Arrow(end=OpenHead(line_color=c, size=10, line_width=1),
                            x_start=pos[body_idx][0][l//2], y_start=pos[body_idx][1][l//2], x_end=pos[body_idx][0][l//2+1], y_end=pos[body_idx][1][l//2+1]))
                if(not print_legend):
                    fig.legend.visible = False
                else:
                    fig.add_layout(fig.legend[0], 'right')

                # ref 
                if(type(idx_ref) == int and idx_ref < len(self.schemes)):
                    pos = np.moveaxis(self.positions[idx_ref], 0, -1)  # [nb_planets,4,nb_steps]
                    for body_idx in range(self.number_of_bodies):
                        fig.line(x=pos[body_idx][0],y=pos[body_idx][1], color = 'black')

                if(type(ncols)==int):
                    grid.append(fig)
                else:
                    show(fig)
        if(type(ncols) == int):
            grid_plot = gridplot(grid, ncols = ncols, merge_tools=True)
            show(grid_plot)

    def plot_energy(self, per_scheme = False, per_body = False, sum_energy = False, max_figs = 10, ncols = None, print_legend = False):
        if(per_body):
            i = 0
            grid = []
            for idx_body in range(0,self.number_of_bodies,max(self.number_of_bodies//max_figs+1,1)):
                i+=1
                if(self.names!=None): title = '{} : '.format(self.names[idx_body])
                else: title = 'Body {}/{}'.format(idx_body,self.number_of_bodies)

                fig = figure(plot_width=980, plot_height=500, match_aspect=False, title=title, \
                    x_axis_label='Fractional change in Energy (%)', y_axis_label='Time (days)',)
                colors = itertools.cycle(palette) 

                for scheme in range(len(self.schemes)):
                    times = np.linspace(0, self.number_of_steps[scheme]*self.dt_in_days[scheme], self.number_of_steps[scheme])
                    label = "{} ; dt = {} days ; theta = {}".format(self.schemes[scheme], self.dt_in_days[scheme], self.theta_list[scheme])
                    fig.line(x=times,y=100*(self.energy_per_body[scheme][idx_body]-self.energy_per_body[scheme][idx_body][0])/self.energy_per_body[scheme][idx_body][0], \
                        color = next(colors), legend_label=label)
                
                if(not print_legend):
                    fig.legend.visible = False
                else:
                    fig.add_layout(fig.legend[0], 'right')

                if(type(ncols)==int):
                    grid.append(fig)
                else:
                    show(fig)
            if(type(ncols) == int):
                grid_plot = gridplot(grid, ncols = ncols, merge_tools=True)
                show(grid_plot)

        if(per_scheme):
           
            for scheme in range(len(self.schemes)): # len(self.schemes) graph
                times = np.linspace(0, self.number_of_steps[scheme]*self.dt_in_days[scheme], self.number_of_steps[scheme])
                title = "{} ; dt = {} days".format(self.schemes[scheme], self.dt_in_days[scheme])
                fig = figure(plot_width=980, plot_height=500, match_aspect=False, title=title, \
                    x_axis_label='Fractional change in Energy (%)', y_axis_label='Time (days)')
                colors = itertools.cycle(palette) 

                for idx_body in range(self.number_of_bodies):
                    if(self.names!=None):
                        fig.line(times,(self.energy_per_body[scheme][idx_body]-self.energy_per_body[scheme][idx_body][0])/self.energy_per_body[scheme][idx_body][0],\
                            color = next(colors), legend_label=self.names[idx_body])
                    else:
                        fig.line(times,100*(self.energy_per_body[scheme][idx_body]-self.energy_per_body[scheme][idx_body][0])/self.energy_per_body[scheme][idx_body][0], \
                            color = next(colors))
                    
                if(not print_legend):
                    fig.legend.visible = False
                else:
                    fig.add_layout(fig.legend[0], 'right')
                if(type(ncols)==int):
                    grid.append(fig)
                else:
                    show(fig)
            if(type(ncols) == int):
                grid_plot = gridplot(grid, ncols = ncols, merge_tools=True)
                show(grid_plot)

        if(sum_energy): # 1 graph
            fig = figure(plot_width=980, plot_height=500, match_aspect=False, title="System's energy", \
                    x_axis_label='Fractional change in Energy (%)', y_axis_label='Time (days)')
            colors = itertools.cycle(palette) 

            for idx_scheme in range(len(self.schemes)):
                times = np.linspace(0, self.number_of_steps[idx_scheme]*self.dt_in_days[idx_scheme], self.number_of_steps[idx_scheme])
                label = "{} ; dt = {} days".format(self.schemes[idx_scheme], self.dt_in_days[idx_scheme])
                fig.line(times, 100*(self.energy_sum[idx_scheme]-self.energy_sum[idx_scheme][0])/self.energy_sum[idx_scheme][0], \
                    color = next(colors), legend_label=label)
                 
            if(not print_legend):
                fig.legend.visible = False
            else:
                fig.add_layout(fig.legend[0], 'right')
            show(fig)
                
    def plot_fn_eval(self, use_log_axis = False, save_fig = False):
        fig, ax = plt.subplots(figsize=(15,10))
        labels = ["{} ; dt = {} days".format(self.schemes[k], self.dt_in_days[k]) for k in range(len(self.schemes))] 
        ax.bar(labels, self.fn_evaluations, log = use_log_axis)
        ax.set_title('Number of function evaluations')
        if(save_fig):
            plt.savefig('fn_eval.png', dpi = 300)
        plt.show()

    # ----------------- Useful for init ------------------ #

    def _compute_energy(self):
        energy_per_body = [np.zeros((self.number_of_bodies,self.number_of_steps[k])) for k in range(len(self.schemes))]
        for k, pos in enumerate(self.positions):
            for step in range(self.number_of_steps[k]):
                for idx_body in range(self.number_of_bodies):
                    energy_per_body[k][idx_body,step]=get_energy_planet(pos[step], self.masses[k], planet_index = idx_body, verbose = False)
        return energy_per_body

    def _compute_sum_energy(self):
        energy_system = [np.zeros((self.number_of_steps[k])) for k in range(len(self.schemes))]
        for idx_scheme in range(len(self.schemes)):
            E_tot = np.zeros((self.number_of_steps[idx_scheme]))
            for idx_body in range(self.number_of_bodies):
                E_tot+=self.energy_per_body[idx_scheme][idx_body]
            energy_system[idx_scheme] = E_tot
        return energy_system

    # ---------------- Use notebook ? ------------------- #

    def isnotebook(self):  # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter
