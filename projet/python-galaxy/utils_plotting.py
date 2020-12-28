
import matplotlib.pyplot as plt
import numpy as np
from utils_energy import get_energy_solar, get_energy_planet # get_center_of_mass
from pprint import pprint

class Plot:
    # here we are plotting for simulations that have the exact same number of time steps.
    # for the exact same bodies (same mass per body, in the same order)
    """
    Class that provides energy computing and plotting utilities. Functions to compute the energies are defined in the module *utils_energy.py*.
    This class takes into input an array or list of positions of format {number of schemes x number of time steps x number of bodies x 2}.
    Note :
        * It is possible to have various lenght in {number of time steps}.
        * It is possible to have various masses (as a function of the scheme used).
    """
    
    def __init__(self, positions, masses, scheme_names, dt_in_days, names = None):
        """ Initialize a plotter by also computing the evolution of energy per body and per schemes and the total energy of the system.
        Various plotting function can be used afterwards to plot : evolution of the system, energy evolutions of the system (per scheme, per body, total energy).

        Args:
            positions (array or list): should be of lenght {number of schemes x number of time steps x number of bodies x 2}.
            masses (list): should be of lenght {number of bodies} or {number of schemes x number of bodies}. If 2D, the i-th scheme received the i-th masses list.
            scheme_names (list of string): should be of lenght {number of schemes}.
            dt_in_days (int or list): if a list or array, the i-th scheme received the i-th dt-in-days.
            names (list of strings, optional): Names of the bodies. Defaults to None.
        """
        self.constant_masses = True if(len(np.array(masses).shape)==1) else False
        self.masses = np.array([masses]*len(scheme_names)) if self.constant_masses else np.array(masses)

        self.constant_dt = True if(type(dt_in_days)==type(1) or type(dt_in_days)==type(1.0)) else False
        self.dt_in_days = np.array([dt_in_days]*len(scheme_names)) if self.constant_dt else np.array(dt_in_days)

        # size = [schemes, nb_steps, nb_planets,2], nb_steps may vary
        self.positions = [np.array(position) for position in positions]
        s = self.positions[0].shape
        self.number_of_bodies = s[1] # makes no sense if it's not the same of everyone
        # something with all(True) etc.
        self.number_of_steps = np.array([pos.shape[0] for pos in self.positions])

        self.schemes = scheme_names
        self.names = names
        
        # TODO : take into account having various masses proportions
        # another possiblity is to never have this case between two simulations inside a 'positions' array
        # and thus to kill the randomness when initializaing the galaxy.

        # init : we compute various data such as the energy etc.
        self.energy_per_body = self._compute_energy() # contains energy ordered by schemes, then body then time step
        self.energy_sum = self._compute_sum_energy() # contains energy ordered by schemes then time step.

    def __repr__(self):
        string = "Plotter :\n\tSchemes : {}\n\tNumber of simulations : {} \n\tNumber of bodies : {} \n\tNumber of time steps : {} \n\tTime step : {} day".format(self.schemes, len(self.schemes), self.number_of_bodies, self.number_of_steps, self.dt_in_days)
        string_ = '\n\tBody names : {}'.format(self.names) if self.names==self.names else ''
        return string + string_

    def plot_positions(self):        
        for k, pos_ in enumerate(self.positions):
            print('Schéma : {}'.format(self.schemes[k]))
            fig, ax = plt.subplots(figsize=(8,5)) 
            pos = np.moveaxis(pos_, 0, -1)  # [nb_planets,2,nb_steps]
            for body_idx in range(self.number_of_bodies):
                if(self.names==self.names):
                    plt.plot(pos[body_idx][0],pos[body_idx][1], label = self.names[body_idx])# [nb_steps, nb_planets,2] -> [nb_planets,2,nb_steps]
                    plt.legend(loc='best')
                else :
                    plt.plot(pos[body_idx][0], pos[body_idx][1])
            plt.show()

    def plot_energy(self, per_scheme = False, per_body = False, sum_energy = False):
        if(per_body):
            for idx_body in range(self.number_of_bodies):
                fig, ax = plt.subplots(figsize=(8,5)) 
                # we don't plot for the sun
                if(self.names==self.names): print('{} : '.format(self.names[idx_body]))
                for scheme in range(len(self.schemes)):
                    times = np.linspace(0, self.number_of_steps[scheme]*self.dt_in_days[scheme], self.number_of_steps[scheme])
                    label = "{} ; dt = {} days".format(self.schemes[scheme], self.dt_in_days[scheme])
                    plt.plot(times,100*(self.energy_per_body[scheme][idx_body]-self.energy_per_body[scheme][idx_body][0])/self.energy_per_body[scheme][idx_body][0], label=label)
                plt.legend(loc='best')
                plt.ylabel('Fractional change in Energy (%)')
                plt.xlabel('Time (days)')
                plt.show()

        if(per_scheme):
            for scheme in range(len(self.schemes)):
                times = np.linspace(0, self.number_of_steps[scheme]*self.dt_in_days[scheme], self.number_of_steps[scheme])
                fig, ax = plt.subplots(figsize=(8,5)) 
                print("{} ; dt = {} days".format(self.schemes[scheme], self.dt_in_days[scheme]))
                for idx_body in range(self.number_of_bodies):
                    if(self.names==self.names):
                        plt.plot(times,(self.energy_per_body[scheme][idx_body]-self.energy_per_body[scheme][idx_body][0])/self.energy_per_body[scheme][idx_body][0], label=self.names[idx_body])
                        plt.legend(loc='best')
                    else:
                        plt.plot(times,100*(self.energy_per_body[scheme][idx_body]-self.energy_per_body[scheme][idx_body][0])/self.energy_per_body[scheme][idx_body][0])
                plt.legend(loc='best')
                plt.ylabel('Fractional change in Energy (%)')
                plt.xlabel('Time (days)')
                plt.show()

        if(sum_energy):
            for idx_scheme in range(len(self.schemes)):
                times = np.linspace(0, self.number_of_steps[idx_scheme]*self.dt_in_days[idx_scheme], self.number_of_steps[idx_scheme])
                label = "{} ; dt = {} days".format(self.schemes[idx_scheme], self.dt_in_days[idx_scheme])
                plt.plot(times, 100*(self.energy_sum[idx_scheme]-self.energy_sum[idx_scheme][0])/self.energy_sum[idx_scheme][0], label=label)
            plt.legend(loc='best')
            plt.ylabel('Fractional change in Energy (%)')
            plt.xlabel('Time (days)')
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