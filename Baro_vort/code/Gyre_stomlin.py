import numpy as np
import dedalus.public as d3
from dedalus.tools.parallel import Sync
import h5py

import time
import logging
logger = logging.getLogger(__name__)

# Numerics Parameters
Nth, Nr = 512, 256
dtype = np.float64

R = 0.5

dealias = 3/2
stop_sim_time = 200
timestepper = d3.RK443
dtype = np.float64

# Physics
eps_s = 0.04

# Bases
coords = d3.PolarCoordinates('th', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nth, Nr), radius=R, dealias=dealias, dtype=dtype)
edge = disk.edge
(th,r)= dist.local_grids(disk) 

# Fields
zeta = dist.Field(bases=disk)
psi = dist.Field(bases=disk)
tau_psi = dist.Field(bases=edge)

beta = dist.Field(bases=disk)
beta['g'] = r*np.sin(th)

wstr = dist.Field(bases=disk)
wstr['g'] = -np.sin(4*np.pi*r*np.sin(th))

# Substitutions
lift = lambda A: d3.Lift(A, disk, -1)
grad = lambda A: d3.Gradient(A)
integ = lambda A: d3.Integrate(A, ('r', 'th'))

KE = d3.integ(grad(psi)@grad(psi))/2
Enstrophy = d3.integ((zeta)**2)/2

# Problem
problem = d3.IVP([zeta, psi, tau_psi], namespace=locals())
problem.add_equation("lap(psi) + lift(tau_psi) = zeta")
problem.add_equation("dt(zeta) + eps_s*zeta = wstr - skew(grad(psi))@grad(beta)")
problem.add_equation("psi(r=R) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

zeta['g'] = 0 # set zeta to zero

# Analysis
snapdata = solver.evaluator.add_file_handler('Gyre_stomlin_snap', sim_dt=10, max_writes=35)
snapdata.add_task(-(-zeta), name='ZETA')
snapdata.add_task(-(-psi), name='PSI')
snapdata.add_task(-(-beta), name='BETA')

diagdata = solver.evaluator.add_file_handler('Gyre_stomlin_diag', sim_dt=1, max_writes=200)
diagdata.add_task(KE, name='KE')
diagdata.add_task(Enstrophy, name='Enstrophy')

# Flow properties
print_freq = 10
flow = d3.GlobalFlowProperty(solver, cadence=print_freq)
flow.add_property(grad(psi)@grad(psi)/2, name='KE')

# Main loop
timestep = 10
delr = R/Nr
try:
    logger.info('Starting main loop')
    solver.step(timestep)
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % print_freq == 0:
            KE_prop = flow.volume_integral('KE')
            logger.info('Iteration=%i, Time=%.3f, dt=%.3e, KE=%.3e' %(solver.iteration, solver.sim_time, timestep, KE_prop))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()