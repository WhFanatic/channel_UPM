import numpy as np
import torch as tc
import fileIO
from tools import Tools_cuda as tool


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

		for t in sorted(set(tsteps1 + tsteps2 + tsteps3)):

			if t in tsteps1: u = fileIO.get_layer(para.fieldpath + 'chan2000.%d.U'%t, j)
			if t in tsteps2: v = fileIO.get_layer(para.fieldpath + 'chan2000.%d.V'%t, j)
			if t in tsteps3: w = fileIO.get_layer(para.fieldpath + 'chan2000.%d.W'%t, j)

			if t in tsteps1: u = tool.spec(u)
			if t in tsteps2: v = tool.spec(v)
			if t in tsteps3: w = tool.spec(w)

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



if __name__ == '__main__':
	from basic2k import DataSetInfo
	from fileIO_statis import Write

	workpath = ''

	para = DataSetInfo('/mnt/TurbNAS/Database/DATABASE_UPM/chan2000/')
	# es2d = EnerSpec2D(para)

	# # compute & save results
	# es2d.calc()
	# es2d.write(workpath)

	# read result
	for yp in (50, 100, 174, 1200):
		j = np.argmin(np.abs(para.yps-yp))
		Euu = tool.flipk(fileIO.get_layer(para.path + 'spectra/Euu.bin', j))
		Evv = tool.flipk(fileIO.get_layer(para.path + 'spectra/Evv.bin', j))
		Eww = tool.flipk(fileIO.get_layer(para.path + 'spectra/Eww.bin', j))
		Euv = tool.flipk(fileIO.get_layer(para.path + 'spectra/Euv.bin', j))

		# write result
		Write.write_es2d(workpath + 'es2d_yp%i.dat'%yp, para, 'DNS2000', Euu, Evv, Eww, Euv)


