import numpy as np

#############################################################@@
class curtiss_model:

    def __init__(self, k):
        self.k = k

    def fcn(self, t, u) :
        k = self.k
        u_dot = k * (np.cos(t) - u)
        return u_dot

    def sol(self, uini, t0, t):
        k = self.k

        c0 = (uini - (k/(k*k + 1)) * (k*np.cos(t0) + np.sin(t0))) * np.exp(k*t0)
        u = (k/(k*k + 1)) * (k*np.cos(t) + np.sin(t)) +  c0 * np.exp(-k*t)
        return u

#############################################################@@
class three_body_model:

    def __init__(self, mu):
        self.mu = mu

    def fcn(self, t, y) :
        y1,y2,y3,y4 = y
        mu = self.mu
        r1 = np.sqrt((y1+mu)*(y1+mu) + y2*y2)
        r2 = np.sqrt((y1-1+mu)*(y1-1+mu) + y2*y2)
        y1_dot = y3
        y2_dot = y4
        y3_dot = y1 + 2*y4 - (1-mu)*(y1+mu)/(r1*r1*r1) - mu*(y1 - 1 + mu)/(r2*r2*r2)
        y4_dot = y2 - 2*y3 - (1-mu)*y2/(r1*r1*r1) - mu*y2/(r2*r2*r2)
        return (y1_dot, y2_dot, y3_dot, y4_dot)

#############################################################@@
def peano_fcn(t, u):
    if t==0. : 
        ret = 0.
    else :
        ret = 4 * ( np.sign(u) * np.sqrt(abs(u)) + max(0, t - abs(u)/t) * np.cos ((np.pi * np.log(t)) / np.log(2)) )
    return ret
