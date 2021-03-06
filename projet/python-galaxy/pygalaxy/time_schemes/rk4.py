import numpy as np

class RK4:
    def __init__(self, dt, nbodies, method, args_method):
        self.dt = dt
        self.method = method
        self.args_method = args_method
        self.k1 = np.zeros((nbodies, 4)) 
        self.k2 = np.zeros((nbodies, 4)) 
        self.k3 = np.zeros((nbodies, 4)) 
        self.k4 = np.zeros((nbodies, 4)) 
        self.tmp = np.zeros((nbodies, 4))
        self.count = 0
        self.count_eval_force = 0


    def init(self, mass, particles):
        pass

    def update(self, mass, particles):
        # k1
        self.count_eval_force += self.method(mass, particles, self.k1, **self.args_method)
        self.tmp[:, :] = particles[:, :4] + self.dt*0.5*self.k1

        # k2
        self.count_eval_force += self.method(mass, self.tmp, self.k2, **self.args_method)
        self.tmp[:, :] = particles[:, :4] + self.dt*0.5*self.k2

        # k3
        self.count_eval_force += self.method(mass, self.tmp, self.k3, **self.args_method)
        self.tmp[:, :] = particles[:, :4] + self.dt*self.k3

        # k4
        self.count_eval_force += self.method(mass, self.tmp, self.k4, **self.args_method)

        particles[:, :] += self.dt/6*(self.k1 + 2*(self.k2+self.k3) + self.k4)

        self.count+=4