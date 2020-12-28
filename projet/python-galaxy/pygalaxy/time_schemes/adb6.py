import numpy as np
from .rk4 import RK4

class ADB6: # lien : https://fr.wikipedia.org/wiki/M%C3%A9thodes_d%27Adams-Bashforth
    def __init__(self, dt, nbodies, method, args_method):
        self.dt = dt
        self.args_method = args_method
        self.method = method
        self.c = [4277.0 / 1440.0,
                 -7923.0 / 1440.0,
                  9982.0 / 1440.0,
                 -7298.0 / 1440.0,
                  2877.0 / 1440.0,
                  -475.0 / 1440.0]
        self.f = np.zeros((6, nbodies, 4)) # comme ordre 6, on a besoin de stocker 6 coefficients à chaque fois.
        self.count = 0

    def init(self, mass, particles): # l'init réalise les 5 premiers pas nécessaires pour le calcul du 6 par Adams Bashforth (ordre 6)
        nbodies = mass.nbodies
        rk4 = RK4(self.dt, nbodies, self.method) # on calcul les premiers coefficients avec rk4 
            # (puisqu'on en a besoin pour cette méthode)

        for i in range(5):
            rk4.update(self, mass, particles)
            self.f[i, :] = rk4.k1 
        
        self.method(mass, particles, self.f[5], **self.args_method)
        self.count += 1 + rk4.count

    def update(self, mass, particles):
        particles[:, :] += self.dt * (self.c[0] * self.f[5] +
                                      self.c[1] * self.f[4] +
                                      self.c[2] * self.f[3] +
                                      self.c[3] * self.f[2] +
                                      self.c[4] * self.f[1] +
                                      self.c[5] * self.f[0])
        self.f = np.roll(self.f, -1, axis=0)
        self.method(mass, particles, self.f[5], **self.args_method) # method est ici l'énergie (dans le cas de l'exemple fournit)
        self.count += 1
        # et fait appel à compute_energy(mass, particles, energy) dans le energy.py du barnes_hut_array
