#!/root/Software/anaconda3/bin/python3
import numpy as np
import torch as tc
import numpy.fft as ft
import fileIO


class EnerSpec2D:
	def __init__(self, para):
		self.para = para

	def calc(self):
		''' compute 2D energy spectra of a whole bulk '''
		para = self.para

		Nyh = para.Ny//2 + para.Ny%2

		self.Euu = np.empty([Nyh, para.Nz, para.Nx//2+1], dtype='>f4')
		self.Evv = np.empty([Nyh, para.Nz, para.Nx//2+1], dtype='>f4')
		self.Eww = np.empty([Nyh, para.Nz, para.Nx//2+1], dtype='>f4')
		self.Euv = np.empty([Nyh, para.Nz, para.Nx//2+1], dtype='>f4')

		for j in range(Nyh):
			print('computing', j)

			Euu1, Evv1, Eww1, Euv1 = self.getes2d(j)
			Euu2, Evv2, Eww2, Euv2 = self.getes2d(para.Ny-1-j)

			self.Euu[j] = .5 * (Euu1 + Euu2)
			self.Evv[j] = .5 * (Evv1 + Evv2)
			self.Eww[j] = .5 * (Eww1 + Eww2)
			self.Euv[j] = .5 * (Euv1 - Euv2)

	def getes2d(self, j):
		''' compute 2D energy spectra of a single layer '''
		para = self.para

		tsteps1 = para.tsteps1
		tsteps2 = para.tsteps2
		tsteps3 = para.tsteps3
		tsteps12 = [t for t in tsteps1 if t in tsteps2]
		tsteps23 = [t for t in tsteps2 if t in tsteps3]
		tsteps13 = [t for t in tsteps1 if t in tsteps3]

		Euu = np.zeros([para.Nz, para.Nx//2+1], dtype='>f4')
		Evv = np.zeros([para.Nz, para.Nx//2+1], dtype='>f4')
		Eww = np.zeros([para.Nz, para.Nx//2+1], dtype='>f4')
		Euv = np.zeros([para.Nz, para.Nx//2+1], dtype='>f4')

		for t in set(tsteps1 + tsteps2 + tsteps3):

			if t in tsteps1: u = fileIO.get_layer(para.fieldpath + 'chan2000.%d.U'%t, j)
			if t in tsteps2: v = fileIO.get_layer(para.fieldpath + 'chan2000.%d.V'%t, j)
			if t in tsteps3: w = fileIO.get_layer(para.fieldpath + 'chan2000.%d.W'%t, j)

			try:
				if t in tsteps1: u = self.spec_cuda(tc.tensor(u.astype(np.float32), device='cuda'))
				if t in tsteps2: v = self.spec_cuda(tc.tensor(v.astype(np.float32), device='cuda'))
				if t in tsteps3: w = self.spec_cuda(tc.tensor(w.astype(np.float32), device='cuda'))
				if t in tsteps1: u = u[0].cpu().numpy() + 1j * u[1].cpu().numpy()
				if t in tsteps2: v = v[0].cpu().numpy() + 1j * v[1].cpu().numpy()
				if t in tsteps3: w = w[0].cpu().numpy() + 1j * w[1].cpu().numpy()
			except:
				if t in tsteps1: u = self.spec(u)
				if t in tsteps2: v = self.spec(v)
				if t in tsteps3: w = self.spec(w)

			if t in tsteps1: u[0,0] = 0
			if t in tsteps2: v[0,0] = 0
			if t in tsteps3: w[0,0] = 0

			if t in tsteps1:  Euu += np.abs(u)**2        / len(tsteps1)
			if t in tsteps2:  Evv += np.abs(v)**2        / len(tsteps2)
			if t in tsteps3:  Eww += np.abs(w)**2        / len(tsteps3)
			if t in tsteps12: Euv += np.real(u.conj()*v) / len(tsteps12)

		return Euu, Evv, Eww, Euv

	def write(self, path):

		info = np.zeros_like(self.Euu[0])
		info.ravel()[:3] = self.Euu.shape[::-1]

		np.concatenate((info, self.Euu), axis=None).astype('>f4').tofile(path + 'Euu.bin')
		np.concatenate((info, self.Evv), axis=None).astype('>f4').tofile(path + 'Evv.bin')
		np.concatenate((info, self.Eww), axis=None).astype('>f4').tofile(path + 'Eww.bin')
		np.concatenate((info, self.Euv), axis=None).astype('>f4').tofile(path + 'Euv.bin')



	@staticmethod
	def spec(q): return ft.ifft(ft.ihfft(q), axis=-2)

	@staticmethod
	def phys(q): return ft.hfft(ft.fft(q, axis=-2)) # Nx must be even

	@staticmethod
	def spec_cuda(q):
		''' do the same spec for GPU tensor q '''
		qr, qi = tc.rfft(q, signal_ndim=2).T
		qr /= tc.prod(tc.tensor(q.shape[-2:]))
		qi /= tc.prod(tc.tensor(q.shape[-2:])) * (-1)
		return tc.stack([qr.T, qi.T]) # torch has no complex type

	@staticmethod
	def flipk(q):
		''' fold all energy to the [:Nz//2+1,:Nx//2+1] range
		    Nx must be even, as required by hft, Nz can be even or odd  '''
		nzcu = int(np.ceil(q.shape[-2]/2))
		nzcd = q.shape[-2]//2
		p = np.copy((q.T[:,:nzcd+1]).T)
		p.T[:,1:nzcu] += q.T[:,:nzcd:-1]
		p.T[1:-1] *= 2
		return p


if __name__ == '__main__':
	from basic2k import DataSetInfo
	from fileIO_statis import Write

	workpath = ''

	para = DataSetInfo('')
	es2d = EnerSpec2D(para)

	# compute & save results
	es2d.calc()
	es2d.write(workpath)

	# read result
	j = np.argmin(np.abs(para.yps-15))
	Euu = EnerSpec2D.flipk(fileIO.get_layer(workpath + 'Euu.bin', j))
	Evv = EnerSpec2D.flipk(fileIO.get_layer(workpath + 'Evv.bin', j))
	Eww = EnerSpec2D.flipk(fileIO.get_layer(workpath + 'Eww.bin', j))
	Euv = EnerSpec2D.flipk(fileIO.get_layer(workpath + 'Euv.bin', j))

	# write result
	Write.write_es2d(workpath + 'ES2D.dat', para, 'DNS2000', Euu, Evv, Eww, Euv)


