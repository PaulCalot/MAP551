import numpy as np

class Euler:
    def __init__(self, dt, nbodies, method, args_method):
        self.dt = dt
        self.method = method
        self.args_method = args_method
        self.k1 = np.zeros((nbodies, 4))

    def init(self, mass, particles):
        pass

    def update(self, mass, particles):
        self.method(mass, particles, self.k1, **self.args_method) # dict
        particles[:, :] += self.dt*self.k1

class Euler_symplectic:
    def __init__(self, dt, nbodies, method, args_method):
        self.dt = dt
        self.method = method
        self.args_method = args_method
        self.k1 = np.zeros((nbodies, 4))

    def init(self, mass, particles):
        pass

    def update(self, mass, particles):
        self.method(mass, particles, self.k1, **self.args_method)
        particles[:, :2] += self.dt*self.k1[:, :2]
        self.method(mass, particles, self.k1, **self.args_method)
        particles[:, 2:] += self.dt*self.k1[:, 2:]