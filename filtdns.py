#!/root/Software/anaconda3/bin/python3
import numpy as np
import torch as tc
from tools import Tools_cuda as tool


class FDNS:
	def __init__(self, para):
		self.para = para

	def filt(self, getlyr, lx, lz, ly=0):
		''' filter the scalar field in 3 directions '''
		para = self.para

		# plane filter: Fourier cut
		i, k = self.getfilt(lx, lz)
		ks = np.roll(range(1-k,k-1), 1-k) # when lx close to 0, this recovers whole kz

		q = np.empty([para.Ny, 2*k-2, i], dtype=np.complex64)

		for j in range(para.Ny):
			print('spec', j)
			q[j] = tool.spec(getlyr(j))[ks,:i]

		# mean profile
		qm = q[:,0,0].real.copy()
		q[:,0,0] = 0

		if not ly: return q, qm

		# wall-normal filter: Box
		qf = np.empty_like(q)
		ys = para.ys

		for j, y in enumerate(ys):
			print('trapz', j)
			ym = max(y-ly/2, ys[0])
			yp = min(y+ly/2, ys[-1])

			jm = self.biSearch(ys, ym)
			jp = self.biSearch(ys, yp)

			qf[j] = np.trapz(q[jm:jp], ys[jm:jp], axis=0) / (ys[jp-1] - ys[jm])

		return qf, qm

	def getfilt(self, lx, lz):
		para = self.para
		
		kx = np.pi / lx # one grid interval can only contain half wave
		kz = np.pi / lz

		i = np.sum(para.kxs<kx)
		k = np.sum(para.kzs<kz)

		return i, k

	@staticmethod
	def biSearch(xs, x):
		''' search for the minimum element in xs that is >= x '''
		im, ip = 0, len(xs)
		while im < ip:
			i = (im + ip) // 2
			if xs[i] < x: im = i+1
			else:         ip = i
		return im


class FDNS_CUDA(FDNS):

	def filt(self, getlyr, lx, lz, ly=0):
		''' filter the scalar field in 3 directions '''
		para = self.para

		# plane filter: Fourier cut
		i, k = self.getfilt(lx, lz)
		ks = tc.tensor(np.roll(range(1-k,k-1), 1-k), device='cuda')

		qr = tc.empty([para.Ny, 2*k-2, i], dtype=tc.float32, device='cuda')
		qi = tc.empty([para.Ny, 2*k-2, i], dtype=tc.float32, device='cuda')

		for j in range(para.Ny):
			print('spec', j)
			ql = tc.tensor(getlyr(j).astype(np.float32), dtype=tc.float32, device='cuda')
			qr[j], qi[j] = self.spec_cuda(ql)[:,ks,:i]

		# mean profile
		qm = qr[:,0,0].cpu().numpy()
		qr[:,0,0] = 0
		qi[:,0,0] = 0

		if not ly: return (qr.cpu().numpy() + 1j * qi.cpu().numpy()), qm

		# wall-normal filter: Box
		qf = np.empty(qr.shape, dtype=np.complex64)
		ys = tc.tensor(para.ys, device='cuda')

		for j, y in enumerate(ys):
			print('trapz', j)
			ym = max(y-ly/2, ys[0])
			yp = min(y+ly/2, ys[-1])

			jm = self.biSearch(ys, ym)
			jp = self.biSearch(ys, yp)

			qf.real[j] = ( tc.trapz(qr[jm:jp], ys[jm:jp], dim=0) / (ys[jp-1] - ys[jm]) ).cpu().numpy()
			qf.imag[j] = ( tc.trapz(qi[jm:jp], ys[jm:jp], dim=0) / (ys[jp-1] - ys[jm]) ).cpu().numpy()

		return qf, qm

	@staticmethod
	def spec_cuda(q):
		''' do the same spec for GPU tensor q '''
		qr, qi = tc.rfft(q, signal_ndim=2).T
		qr /= tc.prod(tc.tensor(q.shape[-2:]))
		qi /= tc.prod(tc.tensor(q.shape[-2:])) * (-1)
		return tc.stack([qr.T, qi.T])
		# check validity
		print(np.sum(np.abs(
			tool.spec(q.cpu().numpy()) - \
			(qr.T.cpu().numpy() + 1j*qi.T.cpu().numpy()))))
		exit()


if __name__ == '__main__':
	from basic2k import DataSetInfo, get_layer

	para = DataSetInfo('/root/data/whn/nas/Database/DATABASE_UPM/chan2000/')
	fdns = FDNS_CUDA(para)

	getlyr = lambda j: get_layer(para.fieldpath + 'chan2000.%d.U'%para.tsteps[0], j)

	uf, um = fdns.filt(getlyr, 100/2000, 100/2000, 100/2000)

	u1 = phys(uf[np.argmin(np.abs(para.yps-174))])
	u2 = phys(uf[np.argmin(np.abs(para.yps-1200))])

	np.savetxt('filt_u_yp174.txt',  u1)
	np.savetxt('filt_u_yp1200.txt', u2)







	# def get2Des(self, yp, tsteps=None):

	# 	para = self.para
	# 	if tsteps is None: tsteps = para.tsteps

	# 	self.Euu = uu = np.zeros([para.Nz//2+1, para.Nx//2+1])
	# 	self.Evv = vv = np.zeros([para.Nz//2+1, para.Nx//2+1])
	# 	self.Eww = ww = np.zeros([para.Nz//2+1, para.Nx//2+1])
	# 	self.Euv = uv = np.zeros([para.Nz//2+1, para.Nx//2+1])

	# 	j = self.biSearch(para.yps, yp)

	# 	for t in tsteps:
	# 		u = get_layer(para.fieldpath + 'chan2000.%d.U'%t, j)
	# 		v = get_layer(para.fieldpath + 'chan2000.%d.V'%t, j)
	# 		w = get_layer(para.fieldpath + 'chan2000.%d.W'%t, j)

	# 		u = spec(u)
	# 		v = spec(v)
	# 		w = spec(w)

	# 		u[0,0] = 0
	# 		v[0,0] = 0
	# 		w[0,0] = 0

	# 		uu += self.__flipk( np.abs(u)**2        ) / len(tsteps)
	# 		vv += self.__flipk( np.abs(v)**2        ) / len(tsteps)
	# 		ww += self.__flipk( np.abs(w)**2        ) / len(tsteps)
	# 		uv += self.__flipk( np.real(u.conj()*v) ) / len(tsteps)

	# def write2Des(self):

	# 	para = self.para

	# 	kxs = rfftfreq(para.Nx, d=para.dx/(2*np.pi))
	# 	kzs = rfftfreq(para.Nz, d=para.dz/(2*np.pi))

	# 	irange = range(1, para.Nx//2+1)
	# 	krange = range(1, para.Nz//2+1)

	# 	header = \
	# 		'Title = "2D energy spectra"\n' + \
	# 		'variables = "%s", "%s", "%s", "%s", "%s", "%s"\n' \
	# 		% (	"log<sub>10</sub>(<greek>l</greek><sub>x</sub><sup>+</sup>)",
	# 			"log<sub>10</sub>(<greek>l</greek><sub>z</sub><sup>+</sup>)",
	# 			"k<sub>x</sub>k<sub>z</sub>E<sub>uu</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
	# 			"k<sub>x</sub>k<sub>z</sub>E<sub>vv</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
	# 			"k<sub>x</sub>k<sub>z</sub>E<sub>ww</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
	# 			"k<sub>x</sub>k<sub>z</sub>E<sub>uv</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",	) + \
	# 		'zone t = "%s", i = %i, j = %i' %( 'Hoyas2008', len(irange), len(krange) )

	# 	data = np.empty([6, len(krange), len(irange)])
	# 	for k in krange:
	# 		for i in irange:
	# 			data[:, krange.index(k), irange.index(i)] = [
	# 				kxs[i],
	# 				kzs[k],
	# 				self.Euu[k,i],
	# 				self.Evv[k,i],
	# 				self.Eww[k,i],
	# 				self.Euv[k,i], ]

	# 	data[2:] *= data[0] * data[1] / (4*np.pi**2 / para.Lx / para.Lz) / para.uc**2
	# 	data[:2] = np.log10(2*np.pi / data[:2] / para.lc)
	# 	data = np.array([np.ravel(temp) for temp in data]).T

	# 	np.savetxt('ES2D.dat', data, header=header, comments='')