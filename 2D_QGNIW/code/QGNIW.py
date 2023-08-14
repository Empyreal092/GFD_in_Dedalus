import numpy as np
import dedalus.public as d3
from dedalus.tools.parallel import Sync
# from matplotlib import pyplot as plt
# import matplotlib.colors as colors
# import cmocean.cm as cmo
import h5py

import time
import logging
logger = logging.getLogger(__name__)

# Numerics Parameters
log_n = 9
L = 10
Lx, Ly = L, L
Nx, Ny = 2**log_n, 2**log_n

dealias = 3/2
stop_sim_time = 3.1+15/(2*np.pi)
timestepper = d3.RK443
dtype = np.float64

#Physical Parameters
kap = 5e-8*((2**(10-log_n))**4)
nu = 5e-8*((2**(10-log_n))**4)

alpha = 0.1
planck = 1

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Fields
q = dist.Field(name='ZETA', bases=(xbasis,ybasis))
psi = dist.Field(name='PSI', bases=(xbasis,ybasis))
tau_psi = dist.Field(name='tau_psi')
phir = dist.Field(name='phir', bases=(xbasis, ybasis))
phii = dist.Field(name='phii', bases=(xbasis, ybasis))

# Substitutions
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
lap = lambda A: d3.Laplacian(A)
integ = lambda A: d3.Integrate(A, ('x', 'y'))

x, y = dist.local_grids(xbasis, ybasis)

# mag2 = lambda f : f * np.conj(f)
J = lambda A, B: dx(A)*dy(B)-dy(A)*dx(B)
l4H = lambda A: lap(lap(A))

qw = lap(phir**2+phii**2)/4 - J(phir, phii)
p_w = -dy(phir)+dx(phii)
vort = lap(psi)

KE = integ(dx(psi)**2+dy(psi)**2)/2
PE = integ((dx(phir)**2+dy(phir)**2+dx(phii)**2+dy(phii)**2)/4*alpha*planck)
Action = integ((phir*phir+phii*phii)/2)

# Problem
problem = d3.IVP([q, psi, tau_psi, phir, phii], namespace=locals())

problem.add_equation("lap(psi) + tau_psi = q - alpha*qw")
problem.add_equation("dt(q) + kap*l4H(q) = - J(psi,q)")
problem.add_equation("integ(psi) = 0")

problem.add_equation("dt(phir) + planck*lap(phii)/2 + nu*l4H(phir) = -J(psi,phir) + phii*lap(psi)/2")
problem.add_equation("dt(phii) - planck*lap(phir)/2 + nu*l4H(phii) = -J(psi,phii) - phir*lap(psi)/2")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
write = solver.load_state("2DEuler_snap/2DEuler_snap_s1.h5", index=32, allow_missing=True)

phir['g'] = 1/np.sqrt(2)
phii['g'] = 1/np.sqrt(2)

# Analysis
snapshots = solver.evaluator.add_file_handler('QGNIW_snap', sim_dt=1/(2*np.pi), max_writes=50)
snapshots.add_task(q, name='PV')
snapshots.add_task(phir, name='phir')
snapshots.add_task(phii, name='phii')
snapshots.add_task(psi, name='psi')
snapshots.add_task(p_w, name='wave_pressure')

diagdata = solver.evaluator.add_file_handler('QGNIW_diag', sim_dt=0.01, max_writes=stop_sim_time*100)
diagdata.add_task(KE, name='KE')
diagdata.add_task(PE, name='PE')
diagdata.add_task(Action, name='Action')

# Flow properties
dt_change_freq = 10
flow_cfl = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow_cfl.add_property(abs(dy(psi)), name='absu')
flow_cfl.add_property(abs(dx(psi)), name='absv')
flow_cfl.add_property(abs(vort), name='absvort')

print_freq = 100
flow = d3.GlobalFlowProperty(solver, cadence=print_freq)
# flow.add_property(abs(-q), name='abs_q')
flow.add_property((-psi*lap(psi))/2, name='KE')
flow.add_property((dx(phir)**2+dy(phir)**2+dx(phii)**2+dy(phii)**2)/4*alpha*planck, name='PE')
flow.add_property((phir*phir+phii*phii)/2, name='Action')

# Main loop
timestep = 1e-7
delx = Lx/Nx; dely = Ly/Ny

sim_time_init = solver.sim_time
try:
    logger.info('Starting main loop')
    solver.step(timestep)
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % dt_change_freq == 0:
            maxU = max(1e-10,flow_cfl.max('absu')); maxV = max(1e-10,flow_cfl.max('absv')); maxvort = max(1e-10,flow_cfl.max('absvort'))
            timestep_CFL = min(delx/maxU,dely/maxV); timestep_img = 1/maxvort
            timestep = max(1e-7, min(timestep_CFL,timestep_img,0.1))*0.1
        if (solver.iteration-1) % print_freq == 0:
            KE_int = flow.volume_integral('KE')
            PE_int = flow.volume_integral('PE')
            Action_int = flow.volume_integral('Action')
            # max_absq = flow.max('abs_q')
            logger.info('Iteration=%i, Time=%.4f, dt=%.3e, KE=%.3f, PE=%.3f, KE+PE=%.3f, Act-50=%.3e' %(solver.iteration, (solver.sim_time-sim_time_init)*2*np.pi, timestep, KE_int, PE_int, KE_int+PE_int, Action_int-50))
            
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

