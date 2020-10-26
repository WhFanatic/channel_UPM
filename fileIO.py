import numpy as np



def get_layer(filename, j):
	''' read data from an x-z plane, index j is 0-based '''
	nx, nz, ny = map(int, np.fromfile(filename, dtype=">f4", count=3))
	return np.fromfile(filename, dtype=">f4", count=nx*nz, offset=4*nx*nz*(j+1)).reshape([nz,nx])

def get_stream(filename, j, k):
	''' read data along an x line, indices j & k are 0-based '''
	nx, nz, ny = map(int, np.fromfile(filename, dtype=">f4", count=3))
	return np.fromfile(filename, dtype=">f4", count=nx, offset=4*nx*(nz*(j+1)+k)).ravel()


def loadtec(filename, fmt=1, skiprows=3):
	data = np.loadtxt(filename, skiprows=skiprows)

	if skiprows == 3:
		nx, ny, nz = 0, 0, 0
	
		with open(filename) as fp:
			line = fp.readline()
			line = fp.readline()
			line = fp.readline()
			for term in line.split(','):
				if term.strip()[0] == 'i': nx = int(term.split('=')[-1])
				if term.strip()[0] == 'j': ny = int(term.split('=')[-1])
				if term.strip()[0] == 'k': nz = int(term.split('=')[-1])

		# 3D data: x, y, z, data
		if nz:
			data = data.T.reshape([-1,nz,ny,nx])
			if fmt==1: return data[0,0,0,:], data[1,0,:,0], data[2,:,0,0], data[3:]
			if fmt==2: return data[0,:,0,0], data[1,0,:,0], data[2,0,0,:], np.array([f.T for f in data[3:]])
		# 2D data: x, y, data
		elif ny:
			data = data.T.reshape([-1,ny,nx])
			if fmt==1: return data[0,0,:], data[1,:,0], data[2:]
			if fmt==2: return data[0,:,0], data[1,0,:], np.array([f.T for f in data[2:]])

	# 1D data: x, data
	return data.T[0], data.T[1:]





