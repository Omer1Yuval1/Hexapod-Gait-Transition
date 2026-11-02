import numpy as np

def natural_frame(X,W):
	
	'''
	Input arguments
		% XYZ is a [3 x Np] matrix of midline coordinates, where Np is the number of midline points, and column are [x,y,z] of midline coordinates.
		% W is a [3 x Np] matrix of normal vectors (not normalised, i.e., scaled by absolute curvature), where Np is the number of midline points, and column are [x,y,z] of curvature vectors.
	'''
	
	gs_tol = 1.0e-12 # Tolerance for gram-schmit.
	
	# find frame
	def findFrame(X):
		nPoints = np.shape(X)[1]
		tau = np.zeros((3,nPoints))
		M1 = np.zeros((3,nPoints))
		M2 = np.zeros((3,nPoints))
		for j in range(nPoints-1): # j = 1 : nPoints-1
			tauJtilde = X[:,j+1] - X[:,j] # X(:,j+1) - X(:,j);
			tau[:,j] = tauJtilde / np.linalg.norm(tauJtilde) # tau(:,j) = tauJtilde / norm(tauJtilde);
			if(j == 0):
				A = np.column_stack( (tau[:,j], np.matrix('1 0 0 ; 0 1 0 ; 0 0 1')) )
				Q = GS(A)
				M1tilde = Q[:,1] # TODO: double-check this.
				M1[:,j] = M1tilde / np.linalg.norm(M1tilde)
				M2[:,j] = np.cross(tau[:,j],M1[:,j])
			else:
				M1tilde = np.add(M1[:,j-1], - (np.matmul(tau[:,j].T, np.transpose([M1[:,j-1]])) * tau[:,j]).T)
				M1[:,j] = M1tilde / np.linalg.norm(M1tilde)
				M2[:,j] = np.cross(tau[:,j],M1[:,j])
		
		return tau,M1,M2
	
	# decompose
	def decompose(M1,M2,W):
		nPoints = np.shape(W)[1]
		m1 = np.zeros((nPoints,1))
		m2 = np.zeros((nPoints,1))
		for j in range(1,nPoints-1): # j=2:nPoints-1
			m1[j] = np.matmul(np.transpose(M1[:,j]), W[:,j])
			m2[j] = np.matmul(np.transpose(M2[:,j]), W[:,j])
		
		return m1,m2
	
	# find normalisation rotation
	def find_theta(m10,m20,M10,M20):
		m12 = np.add(np.power(m10,2), np.power(m20,2))
		j = np.argmax(m12)
		theta = np.arctan2(-m20[j],m10[j])
		
		return theta
	
	# normalise
	def normalised(m10,m20,M10,M20,theta):
		m1 = np.add(np.sin(theta) * m10, + np.cos(theta) * m20)
		m2 = np.add(np.cos(theta) * m10, - np.sin(theta) * m20)
		
		M1 = np.add(np.sin(theta) * M10, + np.cos(theta) * M20)
		M2 = np.add(np.cos(theta) * M10, - np.sin(theta) * M20)
		
		return m1,m2,M1,M2
	
	# gram-schmit
	def GS(A): # Q, R = np.linalg.qr(X)
		m, n = np.shape(A) # [m,n] = size(A);
		Q = np.zeros((m,n))
		for j in range(n): # for j = 1:n
			v = A[:,j] # [n,1]
			for i in range(j): # for i = 1:j-1
				v = np.add(v, - (np.matmul(v.T, np.transpose([Q[:,i]])) * Q[:,i]).T) # v = v - v'*Q(:,i)*Q(:,i);
			
			if(np.linalg.norm(v) > gs_tol):
				Q[:,j] = (v / np.linalg.norm(v)).T
		
		Q = Q[:,~np.all(Q == 0, axis = 0)] # remove zero columns.
		
		return Q
	
	tau,M10,M20 = findFrame(X)
	m10,m20 = decompose(M10,M20,W)
	
	theta = find_theta(m10,m20,M10,M20)
	m1,m2,M1,M2 = normalised(m10,m20,M10,M20,theta)
	
	return M1,M2,m10,m20,m1,m2,tau