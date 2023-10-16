import numpy as np
import dedalus.public as d3
from dedalus.tools.parallel import Sync
import h5py

import time
import logging
logger = logging.getLogger(__name__)

# Numerics Parameters
L = 15; Lx, Ly = L, L
log_n = 9; Nx, Ny = 2**log_n, 2**log_n
dtype = np.float64

dealias = 3/2
stop_sim_time = 200
timestepper = d3.RK443
dtype = np.float64

#Physical Parameters
delx = Lx/Nx
nu8 = 1*delx**8
nu0 = 0.4

xi = 0.4
# xi = 0.6

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Fields
q1 = dist.Field(bases=(xbasis,ybasis) )
q2 = dist.Field(bases=(xbasis,ybasis) )
psi1 = dist.Field(bases=(xbasis,ybasis) )
psi2 = dist.Field(bases=(xbasis,ybasis) )

tau_psi1 = dist.Field()

# Substitutions
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
lap = lambda A: d3.Laplacian(A)
integ = lambda A: d3.Integrate(A, ('x', 'y'))

x, y = dist.local_grids(xbasis, ybasis)

J = lambda A, B: dx(A)*dy(B)-dy(A)*dx(B)
# wavenumber interpretation of lap8
l8H = lambda A: dx(dx(dx(dx(dx(dx(dx(dx(A))))))))+dy(dy(dy(dy(dy(dy(dy(dy(A))))))))

KE1 = integ(dx(psi1)**2+dy(psi1)**2)*0.5
KE2 = integ(dx(psi2)**2+dy(psi2)**2)*0.5

# Problem
problem = d3.IVP([q1, q2, psi1, psi2, tau_psi1], namespace=locals())
problem.add_equation("dt(q1)+dx(q1)+(xi**(-2)+8)*dx(psi1) + nu8*l8H(q1) = -J(psi1,q1)")
problem.add_equation("dt(q2)-dx(q2)+(xi**(-2)-8)*dx(psi2) + nu8*l8H(q2)+nu0*lap(psi2) = -J(psi2,q2)")
# problem.add_equation("dt(q2)-dx(q2)+(xi**(-2)-8)*dx(psi2) + nu8*l8H(q2) = -J(psi2,q2)")
problem.add_equation("lap(psi1)+4*(psi2-psi1)+tau_psi1=q1")
problem.add_equation("lap(psi2)+4*(psi1-psi2)=q2")
problem.add_equation("integ(psi1) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
q1.fill_random('c', seed=42, distribution='normal', scale=1e-2) # Random noise
q1.low_pass_filter(shape=(32, 32)); q1.high_pass_filter(shape=(2, 2))
q2.fill_random('c', seed=1314, distribution='normal', scale=1e-2) # Random noise
q2.low_pass_filter(shape=(32, 32)); q2.high_pass_filter(shape=(2, 2))

# Analysis
snapdata = solver.evaluator.add_file_handler('2LayQG_jets_snap', sim_dt=0.5, max_writes=20)
snapdata.add_task(-(-q1), name='q1')
snapdata.add_task(-(-q2), name='q2')
snapdata.add_task(-(-psi1), name='psi1')
snapdata.add_task(-(-psi2), name='psi2')

diagdata = solver.evaluator.add_file_handler('2LayQG_jets_diag', sim_dt=0.01, max_writes=stop_sim_time*100)
diagdata.add_task(KE1, name='KE1')
diagdata.add_task(KE2, name='KE2')

# Flow properties
dt_change_freq = 10
flow = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow.add_property(abs(dy(psi1)), name='absu1')
flow.add_property(abs(dx(psi1)), name='absv1')
flow.add_property(abs(dy(psi2)), name='absu2')
flow.add_property(abs(dx(psi2)), name='absv2')
# flow.add_property(q1**2, name='ens1')
flow.add_property(dx(psi1)**2+dy(psi1)**2, name='KE1')

# Main loop
timestep = 1e-7
delx = Lx/Nx; dely = Ly/Ny
try:
    logger.info('Starting main loop')
    solver.step(timestep)
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % dt_change_freq == 0:
            maxU = max(1e-10,flow.max('absu1'),flow.max('absu2')); maxV = max(1e-10,flow.max('absv1'),flow.max('absv2'))
            timestep_CFL = min(delx/maxU,dely/maxV)*0.5
            timestep = min(max(1e-5, timestep_CFL), 0.01)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration=%i, Time=%.3f, dt=%.3e, KE1=%.3f' %(solver.iteration, solver.sim_time, timestep, np.sqrt(flow.volume_integral('KE1'))))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
