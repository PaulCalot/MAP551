#!/usr/bin/env python

"""
This program uses the Barnes-Hut algorithm to solve the N-body problem on the
solar system.


Usage:
    solar [options]

Options:
    -R, --render=<render_option>    The typology of render engine to be used. By
                                    default it uses `matplotlib`. The other
                                    option is to use the more fancy `opengl`.
                                    [default: matplotlib]

    --step=<step>                   Simulation step between each render
                                    [default: 5]
"""


import importlib
import numpy as np
import pygalaxy
from docopt import docopt
from pygalaxy.barnes_hut_array import compute_energy
from pygalaxy.naive.energy import compute_energy as compute_energy_naive

def get_scheme(scheme):
    # possibilities : ['RK4', 'ADB6', 'Euler', 'Euler_symplectique','Stormer_verlet','Optimized_815']
    if scheme == 'RK4':
        return pygalaxy.RK4
    elif scheme == 'ADB6':
        return pygalaxy.ADB6
    elif scheme == 'Euler' : 
        return pygalaxy.Euler
    elif scheme == 'Euler_symplectic':
        return pygalaxy.Euler_symplectic 
    elif scheme == 'Stormer_verlet':
        return pygalaxy.Stormer_verlet 
    elif scheme == 'Optimized_815':
        return pygalaxy.Optimized_815 
    else :
        print("Could not find scheme. Using 'Optimized_815' instead.")
        return pygalaxy.Optimized_815
    
class SolarSystem:
    def __init__(self, dt=pygalaxy.physics.day_in_sec, display_step=1, scheme = 'Optimized_815', args_method = {'theta':0.5}):
        self.mass, self.particles = pygalaxy.init_solar_system()
        
        self.time_method = get_scheme(scheme)(dt, self.particles.shape[0],
                                                 compute_energy, args_method)
        self.display_step = display_step

    def next(self, return_pos = False):
        if(return_pos): L = np.zeros((self.display_step,len(self.particles),2))
        for i in range(self.display_step):
            self.time_method.update(self.mass, self.particles)
            if(return_pos):L[i]=self.particles[:,:2]
        if(return_pos): return L
        
    def coords(self):
        return self.particles[:, :2]

    # useless (other than to compare execution time)
    def get_energy(self):
        energy = np.zeros((len(self.particles), 4))
        compute_energy(self.mass, self.particles,energy)
        return energy
    
    def get_energy_naive(self):
        energy = np.zeros((len(self.particles), 4))# 4 because 2 positions, 2 speed, per planet.
        compute_energy_naive(self.mass, self.particles,energy)
        return energy
    
if __name__ == '__main__':
    args = docopt(__doc__)

    display_step = int(args['--step'])
    render_engine = args['--render']

    # Importing the right class for rendering from the right module
    anim_module = importlib.import_module('pygalaxy.'+render_engine)
    Animation = getattr(anim_module, 'Animation')

    sim = SolarSystem(10*pygalaxy.physics.day_in_sec,
                      display_step=display_step)

    bmin = np.min(sim.coords(), axis=0)
    bmax = np.max(sim.coords(), axis=0)
    xmin = -1.25*np.max(np.abs([bmin, bmax]))

    anim = Animation(sim, [xmin, -xmin, xmin, -xmin])
    anim.main_loop()
