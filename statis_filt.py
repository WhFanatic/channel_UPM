import numpy as np
from fileIO import get_layer
from filtdns import FDNS, FDNS_CUDA, spec


class Statis:
	def __init__(self, para, lx, lz=None, ly=None):
		self.para = para
		self.fdns = FDNS_CUDA(para)

		self.lx = lx
		self.lz = lz if lz is not None else lx
		self.ly = ly if ly is not None else ly

	def calc_statis(self, tsteps=None):
		
		para = self.para
		i, k = self.fdns.getfilt(self.lx, self.lz)

		if tsteps is None: tsteps = para.tsteps

		self.Um,  self.Vm,  self.Wm  = (np.zeros(para.Ny, np.float32) for _ in range(3))
		self.R11, self.R22, self.R33 = (np.zeros(para.Ny, np.float32) for _ in range(3))
		self.R12, self.R23, self.R13 = (np.zeros(para.Ny, np.float32) for _ in range(3))
		self.Euu, self.Evv, self.Eww = (np.zeros([para.Ny, 2*k-1, i], np.float32)   for _ in range(3))
		self.Euv, self.Evw, self.Euw = (np.zeros([para.Ny, 2*k-1, i], np.complex64) for _ in range(3))

		for t in tsteps:
			print("Reading statis: tstep", t)

			getu = lambda j: get_layer(para.fieldpath + 'chan2000.%d.U'%t, j)
			getv = lambda j: get_layer(para.fieldpath + 'chan2000.%d.V'%t, j)
			getw = lambda j: get_layer(para.fieldpath + 'chan2000.%d.W'%t, j)

			u, um = self.fdns.filt(getu, self.lx, self.lz, self.ly)
			v, vm = self.fdns.filt(getv, self.lx, self.lz, self.ly)
			w, wm = self.fdns.filt(getw, self.lx, self.lz, self.ly)

			self.Um += um / len(tsteps)
			self.Vm += vm / len(tsteps)
			self.Wm += wm / len(tsteps)

			self.Euu += np.abs(u)**2 / len(tsteps)
			self.Evv += np.abs(v)**2 / len(tsteps)
			self.Eww += np.abs(w)**2 / len(tsteps)
			self.Euv += u.conj() * v / len(tsteps)
			self.Evw += v.conj() * w / len(tsteps)
			self.Euw += u.conj() * w / len(tsteps)

		self.R11[:] = np.sum(self.__flipk(self.Euu), axis=(-1,-2))
		self.R22[:] = np.sum(self.__flipk(self.Evv), axis=(-1,-2))
		self.R33[:] = np.sum(self.__flipk(self.Eww), axis=(-1,-2))
		self.R12[:] = np.sum(self.__flipk(self.Euv.real), axis=(-1,-2))
		self.R23[:] = np.sum(self.__flipk(self.Evw.real), axis=(-1,-2))
		self.R13[:] = np.sum(self.__flipk(self.Euw.real), axis=(-1,-2))

	def flipy(self):
		self.Um[:] = .5 * (self.Um + self.Um[::-1])
		self.Vm[:] = .5 * (self.Vm - self.Vm[::-1])
		self.Wm[:] = .5 * (self.Wm + self.Wm[::-1])

		self.R11[:] = .5 * (self.R11 + self.R11[::-1])
		self.R22[:] = .5 * (self.R22 + self.R22[::-1])
		self.R33[:] = .5 * (self.R33 + self.R33[::-1])
		self.R12[:] = .5 * (self.R12 - self.R12[::-1])
		self.R23[:] = .5 * (self.R23 - self.R23[::-1])
		self.R13[:] = .5 * (self.R13 + self.R13[::-1])

		self.Euu[:] = .5 * (self.Euu + self.Euu[::-1])
		self.Evv[:] = .5 * (self.Evv + self.Evv[::-1])
		self.Eww[:] = .5 * (self.Eww + self.Eww[::-1])
		self.Euv[:] = .5 * (self.Euv - self.Euv[::-1])
		self.Evw[:] = .5 * (self.Evw - self.Evw[::-1])
		self.Euw[:] = .5 * (self.Euw + self.Euw[::-1])

	def flipk(self):
		self.Euu = self.__flipk(self.Euu)
		self.Evv = self.__flipk(self.Evv)
		self.Eww = self.__flipk(self.Eww)
		self.Euv = self.__flipk(self.Euv.real)
		self.Evw = self.__flipk(self.Evw.real)
		self.Euw = self.__flipk(self.Euw.real)

	@staticmethod
	def __flipk(q):
		''' fold all energy to the [:nzc,:nxc] range
		    Nx must be even, as required by hft, Nz can be even or odd  '''
		nzcu = int(np.ceil(q.shape[-2]/2))
		nzcd = q.shape[-2]//2
		p = np.copy((q.T[:,:nzcd+1]).T)
		p.T[:,1:nzcu] += q.T[:,:nzcd:-1]
		p.T[1:-1] *= 2
		return p



