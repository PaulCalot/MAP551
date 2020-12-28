import numpy as np
import pygalaxy
from pygalaxy.init import getOrbitalVelocity
mass_sun = pygalaxy.physics.mass_sun
gamma_si = pygalaxy.physics.gamma_si
pc_in_m = 3.08567758129e16

# link : http://spiff.rit.edu/classes/phys559/lectures/conserved/conserved.html

def get_center_of_mass(particles, masses):
    M = 0
    X, Y = 0, 0
    for particle, mass in zip(particles, masses):
        X+=particle[0]*mass
        Y+=particle[1]*mass
        M+=mass
    return X/M, Y/M

def get_energy_solar(particles, masses, verbose = True):
    
    # Kinetic Energy
    KE = 0
    X, Y = get_center_of_mass(particles, masses)
    for k in range(len(particles)):
        x,y,Fx,Fy = particles[k]
        mass = masses[k]
        vx, vy = getOrbitalVelocity(x, y, mass, X, Y) # in m/s
        KE += 0.5*mass*(vx*vx+vy*vy)
    
    # Gravitational potential energy GPE
    GPE = 0
    for i in range(len(particles)):
        for j in range(len(particles)):
            if(i!=j):
                mass_i = masses[i]
                mass_j = masses[j]
                xi, yi = particles[i][0], particles[i][1]
                xj, yj = particles[j][0], particles[j][1]
                rij = np.sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))
                GPE-= gamma_si*mass_i*mass_j/(rij*pc_in_m)
    return KE+GPE
    
def get_energy_planet(particles, masses, planet_index, verbose = True):
    # Kinetic Energy
    X, Y = get_center_of_mass(particles, masses)
    x,y, = particles[planet_index][0],particles[planet_index][1]
    mass = masses[planet_index]
    vx, vy = getOrbitalVelocity(x, y, mass, X, Y) # in m/s
    KE = 0.5*mass*(vx*vx+vy*vy)

    # Gravitational potential energy GPE
    GPE = 0
    for i in range(len(particles)):
        if(i!=planet_index):
            mass_i = masses[i]
            mass_j = masses[planet_index]
            xi, yi = particles[i][0], particles[i][1]
            xj, yj = particles[planet_index][0], particles[planet_index][1]
            rij = np.sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))
            GPE-= gamma_si*mass_i*mass_j/(rij*pc_in_m)
                
    return KE+GPE