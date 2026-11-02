import numpy as np

def TNB(x,y,z=None):

	if(z is None):
		z = np.zeros(x.shape)
	
	dr = np.array([np.gradient(x) , np.gradient(y) , np.gradient(z)]).T
	
	ds = np.sum(dr**2,axis=1)**(0.5) # Arc length associated with each point. ||dr||.
	
	T = dr / ds[:,None] # Unit tangent vector.
	
	dT = np.array([np.gradient(T[:,0]) , np.gradient(T[:,1]) , np.gradient(T[:,2])]).T # T'(t).
	dTds = dT / ds[:,None]
	
	kappa = np.sum((dTds)**2, axis=1)**(0.5) # Curvature.
	
	N = dTds / kappa[:,None] # Unit normal vector.
	
	B = np.cross(T,N, axisa=1, axisb=1) # Unit bi-normal vector.
	
	dB = np.array([np.gradient(B[:,0]) , np.gradient(B[:,1]) , np.gradient(B[:,2])]).T
	
	tau = np.sum((-dB/ds[:,None]) * N, axis=1) # Torsion.
	
	return T,N,B,kappa,tau