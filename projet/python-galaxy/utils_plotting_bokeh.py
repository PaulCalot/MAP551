
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

# widget
# links :
# https://docs.bokeh.org/en/latest/docs/user_guide/interaction/widgets.html
#from bokeh.models import CustomJS, Select

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
    fontsize_axis = '11pt'
    fontsize_title = '10pt'
    fontsize_legend = '11pt'
    fontsize_major_tick = '11pt'

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
        self.names = names if names != None else ["body {}".format(k+1) for k in range(self.number_of_bodies)]  

        self.fn_evaluations = fn_evaluations
        
        self.theta_list = theta if type(theta) == list else np.array([theta]*len(scheme_names)) # only useful for plotting names

        # init : we compute various data such as the energy etc.
        self.energy_per_body = self._compute_energy() # contains energy ordered by schemes, then body then time step
        self.energy_sum = self._compute_sum_energy() # contains energy ordered by schemes then time step.

        # name of simulation with their parameters
        self.simu_names = ["{} ; dt = {} days ; theta = {}".format(s, dt, t) \
            for s,dt,t in zip(self.schemes, self.dt_in_days, self.theta_list)]

        output_notebook()

    def __repr__(self):
        string = "Plotter :\n\tSchemes : {}\n\tNumber of simulations : {} \n\tNumber of bodies : {} \n\tNumber of time steps : {} \n\tTime step : {} day".format(', '.join(self.schemes), len(self.schemes), self.number_of_bodies, ', '.join([str(o) for o in self.number_of_steps]), ', '.join([str(dt) for dt in self.dt_in_days]))
        string_ = '\n\tBody names : {}'.format(', '.join(self.names)) if self.names!=None else ''
        return string + string_

    def set_default_fig(self, fig):
        fig.xaxis.axis_label_text_font_size = self.fontsize_axis
        fig.yaxis.axis_label_text_font_size = self.fontsize_axis
        fig.title.text_font_size = self.fontsize_title
        fig.legend.label_text_font_size = self.fontsize_legend
        fig.yaxis.major_label_text_font_size = self.fontsize_major_tick
        fig.xaxis.major_label_text_font_size = self.fontsize_major_tick

    def plot_orbit(self, ncols = None, print_legend = False, print_arrow = True, idx_ref = None):
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
                    fig.line(x=pos[body_idx][0],y=pos[body_idx][1], color = c, legend_label = self.names[body_idx], line_width=1.5)
                    l = len(pos[body_idx][0])
                    if(print_arrow):
                        fig.add_layout(Arrow(end=OpenHead(line_color=c, size=10, line_width=1),
                            x_start=pos[body_idx][0][l//2], y_start=pos[body_idx][1][l//2], x_end=pos[body_idx][0][l//2+1], y_end=pos[body_idx][1][l//2+1]))
                if(not print_legend):
                    fig.legend.visible = False
                else:
                    fig.legend.click_policy="hide"
                    fig.add_layout(fig.legend[0], 'right')

                # ref
                if(type(idx_ref) == int and idx_ref < len(self.schemes)):
                    pos = np.moveaxis(self.positions[idx_ref], 0, -1)  # [nb_planets,4,nb_steps]
                    for body_idx in range(self.number_of_bodies):
                        fig.line(x=pos[body_idx][0],y=pos[body_idx][1], color = 'black', line_width=1.5, line_alpha=1)
                self.set_default_fig(fig)
                if(type(ncols)==int):
                    grid.append(fig)
                else:
                    #def update_plot(attr, old, new):  
                    #select = Select(title="Select simulation:", value=self.simu_names[0], options=self.simu_names)
                    #select.on_change('value', update_plot)
                    show(fig)
        if(type(ncols) == int):
            grid_plot = gridplot(grid, ncols = ncols, merge_tools=True)
            show(grid_plot)

    def plot_positions(self, axis = 0, print_legend = False, idx_ref = None): # axis = 0 => x ; axis = 1 => y
        for k, pos_ in enumerate(self.positions):
            if(k != idx_ref):
                pos = np.moveaxis(pos_, 0, -1)  # [nb_planets,4,nb_steps]
                name_axis = 'x' if axis == 0 else 'y'
                fig = figure(plot_width=980, plot_height=500, match_aspect=False, \
                    title="{} ; dt = {} days ; theta = {} ; graphe of {} = f(t)".format(self.schemes[k], self.dt_in_days[k], self.theta_list[k], name_axis), \
                        x_axis_label='t', y_axis_label='x')
                colors = itertools.cycle(palette)
                for body_idx in range(self.number_of_bodies):
                    times = np.linspace(0, self.number_of_steps[k]*self.dt_in_days[k], self.number_of_steps[k])
                    c = next(colors)
                    fig.line(x=times,y=pos[body_idx][axis], color = c, legend_label = self.names[body_idx], line_width=1.5,)

                if(not print_legend):
                    fig.legend.visible = False
                else:
                    fig.legend.click_policy="hide"
                    fig.add_layout(fig.legend[0], 'right')

                # ref
                if(type(idx_ref) == int and idx_ref < len(self.schemes)):
                    pos = np.moveaxis(self.positions[idx_ref], 0, -1)  # [nb_planets,4,nb_steps]
                    times = np.linspace(0, self.number_of_steps[idx_ref]*self.dt_in_days[idx_ref], self.number_of_steps[idx_ref])
                    for body_idx in range(self.number_of_bodies):
                        fig.line(x=times,y=pos[body_idx][axis], color = 'black',line_width=1.5, line_alpha=1)
                self.set_default_fig(fig)
                show(fig)

    def plot_energy(self, per_scheme = False, per_body = False, sum_energy = False, exclude = None, max_figs = 10, ncols = None, print_legend = False):
        excluded_idx = [] if exclude == None else exclude # exclude body if per_scheme selected, else exclude schemes.
        if(per_body):
            grid = []
            for idx_body in range(0,self.number_of_bodies,max(self.number_of_bodies//max_figs+1,1)):
                if(self.names!=None): title = '{} : '.format(self.names[idx_body])
                else: title = 'Body {}/{}'.format(idx_body,self.number_of_bodies)

                fig = figure(plot_width=980, plot_height=500, match_aspect=False, title=title, \
                    y_axis_label='Fractional change in Energy (%)', x_axis_label='Time (days)',)
                colors = itertools.cycle(palette) 

                for scheme in range(len(self.schemes)):
                    if(not scheme in excluded_idx):
                        times = np.linspace(0, self.number_of_steps[scheme]*self.dt_in_days[scheme], self.number_of_steps[scheme])
                        label = "{} ; dt = {} days ; theta = {}".format(self.schemes[scheme], self.dt_in_days[scheme], self.theta_list[scheme])
                        fig.line(x=times,y=100*(self.energy_per_body[scheme][idx_body]-self.energy_per_body[scheme][idx_body][0])/self.energy_per_body[scheme][idx_body][0], \
                            color = next(colors), legend_label=label, line_width=2, line_alpha=1)
                    
                if(not print_legend):
                    fig.legend.visible = False
                else:
                    fig.legend.click_policy="hide"
                    fig.add_layout(fig.legend[0], 'right')

                self.set_default_fig(fig)

                if(type(ncols)==int):
                    grid.append(fig)
                else:
                    show(fig)
            if(type(ncols) == int):
                grid_plot = gridplot(grid, ncols = ncols, merge_tools=True)
                show(grid_plot)

        if(per_scheme):
            grid = []
            for scheme in range(len(self.schemes)): # len(self.schemes) graph
                times = np.linspace(0, self.number_of_steps[scheme]*self.dt_in_days[scheme], self.number_of_steps[scheme])
                title = "{} ; dt = {} days".format(self.schemes[scheme], self.dt_in_days[scheme])
                fig = figure(plot_width=980, plot_height=500, match_aspect=False, title=title, \
                    y_axis_label='Fractional change in Energy (%)', x_axis_label='Time (days)')                
                colors = itertools.cycle(palette)

                for idx_body in range(self.number_of_bodies):
                    if(not idx_body in excluded_idx):
                        fig.line(times,(self.energy_per_body[scheme][idx_body]-self.energy_per_body[scheme][idx_body][0])/self.energy_per_body[scheme][idx_body][0],\
                            color = next(colors), legend_label=self.names[idx_body], line_width=2, line_alpha=1)
                    
                if(not print_legend):
                    fig.legend.visible = False
                else:
                    fig.legend.click_policy="hide"
                    fig.add_layout(fig.legend[0], 'right')

                self.set_default_fig(fig)

                if(type(ncols)==int):
                    grid.append(fig)
                else:
                    show(fig)
            if(type(ncols) == int):
                grid_plot = gridplot(grid, ncols = ncols, merge_tools=True)
                show(grid_plot)

        if(sum_energy): # 1 graph
            self._plot_sum_energy(excluded_schemes_from_plotting = excluded_idx, print_legend = print_legend)

    def _plot_sum_energy(self, excluded_schemes_from_plotting = None, excluded_body_from_computation = None, print_legend = False):
        excluded_idx = [] if excluded_schemes_from_plotting == None else excluded_schemes_from_plotting # exclude body if per_scheme selected, else exclude schemes.
        
        if(excluded_body_from_computation!=None):
            energy = self._compute_sum_energy(exclude = excluded_body_from_computation)
        else:
            energy = self.energy_sum
        
        fig = figure(plot_width=980, plot_height=500, match_aspect=False, title="System's energy", \
                    y_axis_label='Fractional change in Energy (%)', x_axis_label='Time (days)')

        colors = itertools.cycle(palette)
        for idx_scheme in range(len(self.schemes)):
            if(not idx_scheme in excluded_idx):
                times = np.linspace(0, self.number_of_steps[idx_scheme]*self.dt_in_days[idx_scheme], self.number_of_steps[idx_scheme])
                label = "{} ; dt = {} days".format(self.schemes[idx_scheme], self.dt_in_days[idx_scheme])
                fig.line(times, 100*(energy[idx_scheme]-energy[idx_scheme][0])/energy[idx_scheme][0], \
                    color = next(colors), legend_label=label, line_width=2, line_alpha=1)
        
        if(not print_legend):
            fig.legend.visible = False
        else:
            fig.legend.click_policy="hide"
            fig.add_layout(fig.legend[0], 'right')
        self.set_default_fig(fig)
        show(fig)

    def plot_fn_eval(self, use_log_axis = False, save_fig = False):
        plt.rcParams['font.size'] = 15
        fig, ax = plt.subplots(figsize=(15,10))
        labels = ["{} ; dt = {} days".format(self.schemes[k], self.dt_in_days[k]) for k in range(len(self.schemes))] 
        ax.bar(labels, self.fn_evaluations, log = use_log_axis)
        ax.set_title('Number of function evaluations')
        if(save_fig):
            plt.savefig('fn_eval.png', dpi = 300)
        plt.show()

    def plot_energy_pie_chart(self, print_legend = True): # only work for the solar system
        # https://docs.bokeh.org/en/latest/docs/gallery/pie_chart.html
        import pandas as pd
        from bokeh.transform import cumsum
        from math import pi
        from bokeh.palettes import Category20c

        for idx_scheme in range(len(self.schemes)):
            x = {key : abs(energy[-1] - energy[0]) for key, energy in zip(self.names, self.energy_per_body[idx_scheme])}
            data = pd.Series(x).reset_index(name='value').rename(columns={'index':'planet'})
            data['angle'] = data['value']/data['value'].sum() * 2*pi
            data['color'] = palette[:len(x)] # Category20c[len(x)]
            
            fig = figure(plot_height=500, match_aspect=True, title="Pie chart of the energy - {}".format(self.simu_names[idx_scheme]),\
                tooltips="@planet: @angle", x_range=(-0.5, 0.5))
                # toolbar_location='right', tools="hover" 
            fig.wedge(x=0, y=1, radius=0.40, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'), \
                line_color="white", fill_color='color', legend_field='planet', source=data)

            self.set_default_fig(fig)
            fig.axis.axis_label=None
            fig.axis.visible=False
            fig.legend.visible = print_legend
            fig.grid.grid_line_color = None

            show(fig)
        
    # ----------------- Useful for init ------------------ #

    def _compute_energy(self):
        energy_per_body = [np.zeros((self.number_of_bodies,self.number_of_steps[k])) for k in range(len(self.schemes))]
        for k, pos in enumerate(self.positions):
            for step in range(self.number_of_steps[k]):
                for idx_body in range(self.number_of_bodies):
                    energy_per_body[k][idx_body,step]=get_energy_planet(pos[step], self.masses[k], planet_index = idx_body, verbose = False)
        return energy_per_body

    def _compute_sum_energy(self, exclude = None):
        excluded_idx = [] if exclude==None else exclude
        energy_system = [np.zeros((self.number_of_steps[k])) for k in range(len(self.schemes))]
        for idx_scheme in range(len(self.schemes)):
            E_tot = np.zeros((self.number_of_steps[idx_scheme]))
            for idx_body in range(self.number_of_bodies):
                if(not idx_body in excluded_idx):
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
