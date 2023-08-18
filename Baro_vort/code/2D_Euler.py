import numpy as np
import dedalus.public as d3
from dedalus.tools.parallel import Sync
import h5py

import time
import logging
logger = logging.getLogger(__name__)

# Numerics Parameters
L = 15; Lx, Ly = L, L
log_n = 10; Nx, Ny = 2**log_n, 2**log_n
dtype = np.float64

dealias = 3/2
stop_sim_time = 30
timestepper = d3.RK443
dtype = np.float64

#Physical Parameters
delx = Lx/Nx
nu8 = 0.5*delx**8

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Fields
zeta = dist.Field(bases=(xbasis,ybasis) )
psi = dist.Field(bases=(xbasis,ybasis) )
tau_psi = dist.Field()

# Substitutions
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
lap = lambda A: d3.Laplacian(A)
integ = lambda A: d3.Integrate(A, ('x', 'y'))

x, y = dist.local_grids(xbasis, ybasis)

J = lambda A, B: dx(A)*dy(B)-dy(A)*dx(B)
# wavenumber interpretation of lap8
l8H = lambda A: dx(dx(dx(dx(dx(dx(dx(dx(A))))))))+dy(dy(dy(dy(dy(dy(dy(dy(A))))))))

KE = integ(dx(psi)**2+dy(psi)**2)/2
Enstrophy = integ(zeta**2)/2

# Problem
problem = d3.IVP([zeta, psi, tau_psi], namespace=locals())
problem.add_equation("lap(psi) + tau_psi = zeta")
problem.add_equation("dt(zeta) + nu8*l8H(zeta) = - J(psi,zeta)")
problem.add_equation("integ(psi) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
zeta.fill_random('c', distribution='normal', scale=0.14*2*np.pi) # Random noise
# keep the initial forcing near the same scale
zeta.low_pass_filter(shape=(32, 32)); zeta.high_pass_filter(shape=(28, 28))

# Analysis
snapdata = solver.evaluator.add_file_handler('2DEuler_snap', sim_dt=0.1, max_writes=50)
snapdata.add_task(-(-zeta), name='ZETA')
snapdata.add_task(-(-psi), name='PSI')

diagdata = solver.evaluator.add_file_handler('2DEuler_diag', sim_dt=0.01, max_writes=stop_sim_time*100)
diagdata.add_task(KE, name='KE')
diagdata.add_task(Enstrophy, name='Enstrophy')

# Flow properties
dt_change_freq = 10
flow = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow.add_property(abs(dy(psi)), name='absu')
flow.add_property(abs(dx(psi)), name='absv')
flow.add_property(-psi*zeta/2, name='KE')

# Main loop
timestep = 1e-7
delx = Lx/Nx; dely = Ly/Ny
try:
    logger.info('Starting main loop')
    solver.step(timestep)
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % dt_change_freq == 0:
            maxU = max(1e-10,flow.max('absu')); maxV = max(1e-10,flow.max('absv'))
            timestep_CFL = min(delx/maxU,dely/maxV)*0.5
            timestep = min(max(1e-5, timestep_CFL), 1)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration=%i, Time=%.3f, dt=%.3e, KE=%.3f' %(solver.iteration, solver.sim_time, timestep, flow.volume_integral('KE')))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

