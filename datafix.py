#!/root/Software/anaconda3/bin/python3
import numpy as np
from basic2k import DataSetInfo


if __name__ == '__main__':

	para = DataSetInfo('')

	data = np.zeros(para.Nx*para.Nz, dtype=">f4")
	data[:3] = para.Nx, para.Nz, para.Ny

	with open('U.bin', 'wb') as fp:
		fp.write(data.tobytes())
		for j in range(para.Ny):
			print('writing', j)
			data[:] = np.fromfile('fields/chan2000.870.U', dtype=">f4", count=para.Nx*para.Nz, offset=4*para.Nx*para.Nz*(j+1))
			fp.write(data.tobytes())

