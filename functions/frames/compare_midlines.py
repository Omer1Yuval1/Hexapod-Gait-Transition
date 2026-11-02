import numpy as np
from functions.frames.TNB import TNB
from functions.frames.natural_frame import natural_frame

from similaritymeasures import frechet_dist

def compare_midlines(midline_0=None, midline_1=None, scaling_factor=None):
    
    df = frechet_dist(midline_0, midline_1)
    score = df
    M1_0 = M2_0 = m1_0 = m2_0 = M1_1 = M2_1 = m1_1 = m2_1 = 0
    midline_0 = np.swapaxes(midline_0, 0, 1)
    midline_1 = np.swapaxes(midline_1, 0, 1)
    # print(midline_0.shape)
    
    '''
    x0 = midline_0[:,0]
    y0 = midline_0[:,1]
    z0 = midline_0[:,2]
    
    x1 = midline_1[:,0]
    y1 = midline_1[:,1]
    z1 = midline_1[:,2]
    
    # Get normal vectors and curvature
    _, N0, _, kappa0, _ = TNB(x0,y0,z0)
    _, N1, _, kappa1, _ = TNB(x1,y1,z1)
    
    midline_0 = np.swapaxes(midline_0, 0, 1)
    midline_1 = np.swapaxes(midline_1, 0, 1)
    
    N0 = np.swapaxes(N0, 0, 1)
    N1 = np.swapaxes(N1, 0, 1)
    
    M1_0, M2_0, m10_0, m20_0, m1_0, m2_0, _ = natural_frame(midline_0, N0 * kappa0)
    M1_1, M2_1, m10_1, m20_1, m1_1, m2_1, _ = natural_frame(midline_1, N1 * kappa1)
    
    # m1_1_normalized = m1_1 / np.abs(m1_1).max()
    # m2_1_normalized = m2_1 / np.abs(m2_1).max()
    
    # Find the difference between the normalised m1 and m2 between the two curves
    # Note that m1 and m2 cannot be nomralized within [-1,1], otherwise the optimisation will only reproduce their shape (and not their magnitude).
    # m1_score = np.linalg.norm((m1_0 / np.abs(m1_0).max()) - (m1_1 / np.abs(m1_1).max()))
    # m2_score = np.linalg.norm((m2_0 / np.abs(m2_0).max()) - (m2_1 / np.abs(m2_1).max()))
    
    m1_1 *= scaling_factor
    m2_1 *= scaling_factor
    
    dm1_0 = np.linalg.norm(m1_0 - m1_1)
    dm1_1 = np.linalg.norm(m1_0 - m1_1[::-1]) # x-flip.
    dm1_2 = np.linalg.norm(m1_0 + m1_1) # y-flip.
    dm1_3 = np.linalg.norm(m1_0 + m1_1[::-1]) # xy-flip.
    
    dm1 = (dm1_0, dm1_1, dm1_2, dm1_3)
    
    dm2_0 = np.linalg.norm(m2_0 - m2_1)
    dm2_1 = np.linalg.norm(m2_0 - m2_1[::-1])
    dm2_2 = np.linalg.norm(m2_0 + m2_1)
    dm2_3 = np.linalg.norm(m2_0 + m2_1[::-1])
    dm2 = (dm2_0, dm2_1, dm2_2, dm2_3)
    
    Imin_m1 = np.argmin(dm1)
    Imin_m2 = np.argmin(dm2)
    
    m1_score = dm1[Imin_m1]
    m2_score = dm2[Imin_m2]
    
    match Imin_m1:
        case 1:
            m1_1 = m1_1[::-1]
        case 2:
            m1_1 = -m1_1
        case 3:
            m1_1 = -m1_1[::-1]
            
    match Imin_m2:
        case 1:
            m2_1 = m2_1[::-1]
        case 2:
            m2_1 = -m2_1
        case 3:
            m2_1 = -m2_1[::-1]
    
    W = [1,1]
    score = (W[0]*m1_score + W[1]*m2_score) / np.sum(np.abs(W))
    '''
    
    return score, midline_0, midline_1, M1_0, M2_0, m1_0, m2_0, M1_1, M2_1, m1_1, m2_1 # Return normalized m1 and m2.
    # return midline_0, midline_1, M1_0, M2_0, m10_0, m20_0, M1_1, M2_1, m10_1, m20_1 # Return un-normalized m1 and m2.