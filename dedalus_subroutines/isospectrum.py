import numpy as np

def isospectrum(q_mag2d):
    N = q_mag2d.shape[0]
    q_spec = np.zeros(int(N/2))
    kx2, ky2 = np.meshgrid(range(N), range(N))
    
    for ki in range(0,int(N/2)):
        mask = (np.floor( (np.sqrt(kx2**2+ky2**2)+1)/2 )) == ki
        mask = mask*1/4; mask[0,:] *= 2; mask[:,0] *= 2
        q_spec[ki] = (mask*q_mag2d).sum()
    
    return q_spec