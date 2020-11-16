import numpy as np


class turing_model_BZ(object):

    def __init__(self, a, b, d1, d2, xmin, xmax, nx) :
        self.a = a
        self.b = b
        self.d1 = d1
        self.d2 = d2
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.dx = (xmax-xmin)/(nx+1)

    def fcn_radau(self, n, t, y, ydot, rpar, ipar):
        a = self.a
        b = self.b
        d1 = self.d1
        d2 = self.d2
        nx = self.nx
        dx = self.dx
        d1overdxdx = d1/(dx*dx)
        d2overdxdx = d2/(dx*dx)

        ui   = y[0]
        vi   = y[1]
        uip1 = y[2]
        vip1 = y[3]
        ydot[0] = d1overdxdx*(-2*ui + 2*uip1) + a - (b+1)*ui + ui*ui*vi
        ydot[1] = d2overdxdx*(-2*vi + 2*vip1) + b*ui - ui*ui*vi

        for ix in range(1, nx-1):
            irow = ix*2
            
            uim1 = y[irow-2]  
            vim1 = y[irow-1]  
            ui   = y[irow]  
            vi   = y[irow+1]  
            uip1 = y[irow+2]  
            vip1 = y[irow+3]  
            ydot[irow]   = d1overdxdx*(uim1 -2*ui + uip1) + a - (b+1)*ui + ui*ui*vi
            ydot[irow+1] = d2overdxdx*(vim1 -2*vi + vip1) + b*ui - ui*ui*vi

        uim1 = y[2*nx -4]
        vim1 = y[2*nx -3]
        ui   = y[2*nx -2]
        vi   = y[2*nx -1]
        ydot[2*nx-2] = d1overdxdx*(2*uim1 -2*ui) + a - (b+1)*ui + ui*ui*vi
        ydot[2*nx-1] = d2overdxdx*(2*vim1 -2*vi) + b*ui - ui*ui*vi
