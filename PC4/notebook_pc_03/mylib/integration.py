import numpy as np
from scipy.optimize import fsolve, root
from scipy.integrate import ode

#################################################################
class ode_result:
    def __init__(self, y, t):
        self.y = y
        self.t = t

#################################################################
def forward_euler(tini, tend, nt, yini, fcn):

    dt = (tend-tini) / (nt-1)
    t = np.linspace(tini, tend, nt)
    
    yini_array = np.array(yini)
    neq = yini_array.size

    y = np.zeros((neq, nt), order='F')
    y[:,0] = yini_array

    for it, tn  in enumerate(t[:-1]):
        yn = y[:,it]
        y[:,it+1] = yn + dt*np.array(fcn(tn, yn))

    return ode_result(y, t)

#################################################################
def backward_euler(tini, tend, nt, yini, fcn):

    dt = (tend-tini) / (nt-1)
    t = np.linspace(tini, tend, nt)

    yini_array = np.array(yini)
    neq = yini_array.size

    y = np.zeros((neq, nt), order='F')
    y[:,0] = yini_array

    def g(uip1, *args):
        uip, tip1 = args
        return uip1 - uip - dt*np.array(fcn(tip1, uip1))

    for it, tn  in enumerate(t[:-1]):
        yn = y[:,it]
        y0 = yn + dt*np.array(fcn(tn, yn))
        # solve y[:,it+1] - y[:,it] - dt * fcn(tini + (it+1)*dt, y[:,it+1]) = 0
        sol = root(g, y0, (yn, tn+dt))
        y[:,it+1] = sol.x

    return ode_result(y, t)
