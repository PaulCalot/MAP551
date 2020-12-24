import numpy as np
import pygalaxy
from pygalaxy.init import getOrbitalVelocity
mass_sun = pygalaxy.physics.mass_sun
gamma_si = pygalaxy.physics.gamma_si
pc_in_m = 3.08567758129e16
def get_energy_solar(particles, mass, index_sun=0, verbose = True):
    E = 0
    xs,ys, Fx, Fy = particles[index_sun]
    for k in range(len(particles)):
        if(k!=index_sun):
            x,y,Fx,Fy = particles[k]
            # distance in parsec
            r = np.sqrt((x-xs)*(x-xs)+(y-ys)*(y-ys))
            if(r!=0.0):
                vx, vy = getOrbitalVelocity(xs, ys, mass_sun, x, y)
                vx, vy = vx, vy
                m = mass[k]
                Ec = 0.5*m*(vx*vx+vy*vy)
                Ep = - gamma_si * mass_sun * m / (r*pc_in_m)
                Eplanet = Ec+Ep
                if(verbose): print("E = {} = {} + {}".format(Eplanet,Ec,Ep))
                E+=Eplanet
    return E
    
def get_energy_planet(xp, yp, mass, xs, ys):
    r = np.sqrt((xp-xs)*(xp-xs)+(yp-ys)*(yp-ys))
    if(r!=0.0):
        vx, vy = getOrbitalVelocity(xs, ys, mass_sun, xp, yp)
        #vx, vy = vx, vy
        m = mass
        Ec = 0.5*m*(vx*vx+vy*vy)
        Ep = - gamma_si * mass_sun * m / (r*pc_in_m)
        Eplanet = Ec+Ep
    return Eplanet