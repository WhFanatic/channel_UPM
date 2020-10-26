#!/root/Software/anaconda3/bin/python3
import os
import torch as tc
import numpy as np
import numpy.fft as ft
from diff import Schemes, Matrices


class Pressure:
	def __init__(self, para):
		self.para = para

	def get_rhs(self, getu, getv, getw, temppath='temp/'):
		''' RHS is diveded into 3 parts according to the order of wall-normal derivative:
			q0 = kx^2 F(uu) + kz^2 F(ww) + 2kxkz F(uw),
			q1 = 2i [kx F(uv) + kz F(vw)],
			q2 = - F(vv),
			and the Poisson equation in Fourier space is re-written as
			(Dyy-k^2) F(p) = q0 + Dy q1 + Dyy q2 '''
		Ny = self.para.Ny
		Nz = self.para.Nz
		dz = self.para.dz
		kxs = self.para.kxs
		kzs = ft.fftfreq(Nz, d=dz/(2*np.pi)).reshape([-1,1])

		os.system('mkdir ' + temppath)

		for j in range(Ny):
			print('%.2f%%'%(100.*j/Ny))

			u = getu(j)
			v = getv(j)
			w = getw(j)

			fuu, fvv, fww, fuv, fvw, fwu = Pressure.spec(
				np.array([u**2, v**2, w**2, u*v, v*w, w*u]))

			q0 = kxs**2 * fuu + kzs**2 * fww + 2*kxs*kzs * fwu
			q1 = 2j * (kxs * fuv + kzs * fvw)
			q2 = -fvv

			q0.astype(np.complex64).tofile(temppath + 'q0_%i'%j)
			q1.astype(np.complex64).tofile(temppath + 'q1_%i'%j)
			q2.astype(np.complex64).tofile(temppath + 'q2_%i'%j)

	def poisson(self, temppath='temp/'):
		''' solve the Poisson equation which is
			further altered to reduce wall-normal differencing:
			(Dyy-k^2) [F(p)-q2] = q0 + Dy q1 + k^2 q2 '''
		Nxc= self.para.Nx//2 + 1
		Ny = self.para.Ny
		Nz = self.para.Nz
		dz = self.para.dz
		ys = self.para.ys
		kxs = self.para.kxs
		kzs = ft.fftfreq(Nz, d=dz/(2*np.pi))

		fp = np.empty([Ny, Nz, Nxc], dtype=np.complex64)
		
		## differencing coefficients
		ones = np.ones(Ny)

		alf1, bet1, a1, b1, c1, d1, e1 = Schemes.compact6_coef(ys, 1)
		alf2, bet2, a2, b2, c2, d2, e2 = Schemes.compact6_coef(ys, 2)

		# Neumann BC coefficients for 2nd order integration
		c2[0] =  (ys[2]-ys[0])**2                   / ((ys[1]-ys[0]) * (ys[2]-ys[0]) * (ys[2]-ys[1]))
		d2[0] = -(ys[1]-ys[0])**2                   / ((ys[1]-ys[0]) * (ys[2]-ys[0]) * (ys[2]-ys[1]))
		e2[0] = ((ys[1]-ys[0])**2-(ys[2]-ys[0])**2) / ((ys[1]-ys[0]) * (ys[2]-ys[0]) * (ys[2]-ys[1]))

		c2[-1] =  (ys[-3]-ys[-1])**2                     / ((ys[-2]-ys[-1]) * (ys[-3]-ys[-1]) * (ys[-3]-ys[-2]))
		b2[-1] = -(ys[-2]-ys[-1])**2                     / ((ys[-2]-ys[-1]) * (ys[-3]-ys[-1]) * (ys[-3]-ys[-2]))
		a2[-1] = ((ys[-2]-ys[-1])**2-(ys[-3]-ys[-1])**2) / ((ys[-2]-ys[-1]) * (ys[-3]-ys[-1]) * (ys[-3]-ys[-2]))

		## solve for each column
		for k, kz in enumerate(kzs):
			print('%.2f%%'%(100.*k/Nz))

			q0 = np.array([np.fromfile(temppath + 'q0_%i'%j, dtype=np.complex64, count=Nxc, offset=8*Nxc*k) for j in range(Ny)])	
			q1 = np.array([np.fromfile(temppath + 'q1_%i'%j, dtype=np.complex64, count=Nxc, offset=8*Nxc*k) for j in range(Ny)])	
			q2 = np.array([np.fromfile(temppath + 'q2_%i'%j, dtype=np.complex64, count=Nxc, offset=8*Nxc*k) for j in range(Ny)])			

			# solve for Dy q1
			q1y = np.empty_like(q1)

			q1y[0] = c1[0]*q1[0] + d1[0]*q1[1] + e1[0]*q1[2]
			q1y[1] = b1[1]*q1[0] + c1[1]*q1[1] + d1[1]*q1[2] + e1[1]*q1[3]
			q1y[-1] = c1[-1]*q1[-1] + b1[-1]*q1[-2] + a1[-1]*q1[-3]
			q1y[-2] = d1[-2]*q1[-1] + c1[-2]*q1[-2] + b1[-2]*q1[-3] + a1[2]*q1[-4]
			q1y[2:-2].T[:] = a1[2:-2] * q1[:-4].T + \
							b1[2:-2] * q1[1:-3].T + \
							c1[2:-2] * q1[2:-2].T + \
							d1[2:-2] * q1[3:-1].T + \
							e1[2:-2] * q1[4:].T
			
			q1y = Pressure.chasing3(alf1, ones, bet1, q1y)

			# solve for F(p) (note: f''+af=g with Mf=Nf'' --> (M+aN)f=Ng)
			k2 = kxs**2 + kz**2

			rhs = q0 + q1y + k2 * q2

			phi = np.empty_like(rhs)
			phi[1:-1].T[:] = alf2[1:-1] * rhs[:-2].T + \
										  rhs[1:-1].T + \
							 bet2[1:-1] * rhs[2:].T
			phi[[0,-1]] = 0 # homo-Neumann BC

			a_ = a_ if 'a_' in locals() else a2.repeat(Nxc).reshape([-1,Nxc])
			b_ = b2.reshape([-1,1]) - k2 * alf2.reshape([-1,1])
			c_ = c2.reshape([-1,1]) - k2
			d_ = d2.reshape([-1,1]) - k2 * bet2.reshape([-1,1])
			e_ = e_ if 'e_' in locals() else e2.repeat(Nxc).reshape([-1,Nxc])

			c_[0], d_[0], e_[0] = c2[0], d2[0], e2[0] # homo-Neumann BC
			a_[-1],b_[-1],c_[-1]= a2[-1],b2[-1],c2[-1] # homo-Neumann BC

			phi = Pressure.chasing5(a_,b_,c_,d_,e_,phi) # F(p)-q2
			phi += q2

			fp[:,k] = phi

		fp.tofile(temppath + 'fp')
		return fp


	@staticmethod
	def spec(q):
		''' do the same spec on GPU '''
		p = tc.tensor(q.astype(np.float32), device='cuda')
		pr, pi = tc.rfft(p, signal_ndim=2).T
		pr /= tc.prod(tc.tensor(p.shape[-2:]))
		pi /= tc.prod(tc.tensor(p.shape[-2:])) * (-1)
		return pr.T.cpu().numpy() + 1j * pi.T.cpu().numpy()

	@staticmethod
	def phys(q):
		''' do the same phys on GPU '''
		nz, nxc = q.shape[-2:]
		pr = tc.tensor(q.real.astype(np.float32), device='cuda')
		pi = tc.tensor(q.imag.astype(np.float32), device='cuda') * (-1)
		p  = tc.irfft(tc.stack([pr.T, pi.T]).T, signal_ndim=2, signal_sizes=[nz, (nxc-1)*2])
		p *= tc.prod(tc.tensor(p.shape[-2:]))
		return p.cpu().numpy()

	@staticmethod
	def chasing3(a_, b_, c_, d_):
		''' alternate direction method for tri-diag equation on GPU,
			dimensions other than 0 are batch dimensions for d '''
		a = tc.tensor(a_.astype(np.float32), device='cuda')
		b = tc.tensor(b_.astype(np.float32), device='cuda')
		c = tc.tensor(c_.astype(np.float32), device='cuda')
		d = tc.stack([ # treat real & imag parts of d as a trailing batch dimension
			tc.tensor(d_.real.astype(np.float32), device='cuda').T,
			tc.tensor(d_.imag.astype(np.float32), device='cuda').T]).T

		N = len(a)
		if not (len(b)==len(c)==len(d)==N): exit()

		r = tc.empty_like(a)
		y = tc.empty_like(d)
		q = tc.empty_like(d)

		r[0] = b[0]
		y[0] = d[0]
		for j in range(1,N):
			l    = a[j] / r[j-1]
			r[j] = b[j] - l*c[j-1]
			y[j] = d[j] - l*y[j-1]
			
		q[-1] = y[-1] / r[-1]
		for j in range(N-1)[::-1]:
			q[j] = (y[j] - c[j]*q[j+1]) / r[j]

		return q.T[0].T.cpu().numpy() + 1j * q.T[1].T.cpu().numpy()

	@staticmethod
	def chasing5(a_, b_, c_, d_, e_, f_):
		''' alternate direction method for penta-diag equation on GPU,
			dimensions other than 0 are batch dimensions for f,
			a~e can be 1D or same size as f '''
		# a b |c d e    |      zt gm |af            |     |1  bt qt      |
		#   a |b c d e  |         zt |gm af         |     |   1  bt qt   |
		#     |a b c d e|    ->      |zt gm af      |  *  |      1  bt qt|
		#     |  a b c d| e          |   zt gm af   |     |         1  bt| qt
		#     |    a b c| d e        |      zt gm af|     |            1 | bt qt
		a = tc.tensor(a_.astype(np.float32), device='cuda')
		b = tc.tensor(b_.astype(np.float32), device='cuda')
		c = tc.tensor(c_.astype(np.float32), device='cuda')
		d = tc.tensor(d_.astype(np.float32), device='cuda')
		e = tc.tensor(e_.astype(np.float32), device='cuda')
		f = tc.stack([ # treat real & imag parts of d as a trailing batch dimension
			tc.tensor(f_.real.astype(np.float32), device='cuda').T,
			tc.tensor(f_.imag.astype(np.float32), device='cuda').T]).T
		
		N  = len(a)
		if not (len(b)==len(c)==len(d)==len(e)==len(f)==N): exit()

		bt = tc.empty_like(a)
		qt = tc.empty_like(a)
		y  = tc.empty_like(f)
		q  = tc.empty_like(f)

		af0   = c[0]
		bt[0] = d[0] / af0
		gm    = b[1]
		af1   = c[1] - gm * bt[0]

		y[0].T[:] =  f[0].T / af0.T
		y[1].T[:] = (f[1].T - gm.T * y[0].T) / af1.T
		
		# direction 1
		for j in range(2,N):
			qt[j-2] = e[j-2] / af0
			bt[j-1] = (d[j-1] - gm * qt[j-2]) / af1
			zt = a[j]
			gm = b[j] - zt * bt[j-2]
			af = c[j] - zt * qt[j-2] - gm * bt[j-1]

			y[j].T[:] = (f[j].T - zt.T * y[j-2].T - gm .T* y[j-1].T) / af.T

			af0, af1 = af1, af

		# direction 2
		q[-1] = y[-1]
		q[-2].T[:] = y[-2].T - bt[-2].T * q[-1].T
		for j in range(N-2)[::-1]:
			q[j].T[:] = y[j].T - qt[j].T * q[j+2].T - bt[j].T * q[j+1].T

		return q.T[0].T.cpu().numpy() + 1j * q.T[1].T.cpu().numpy()




def flipk(q):
	''' fold all energy to the [:nzc,:nxc] range
	    Nx must be even, as required by hft, Nz can be even or odd  '''
	nzcu = int(np.ceil(q.shape[-2]/2))
	nzcd = q.shape[-2]//2
	p = np.copy((q.T[:,:nzcd+1]).T)
	p.T[:,1:nzcu] += q.T[:,:nzcd:-1]
	p.T[1:-1] *= 2
	return p


if __name__ == '__main__':

	para = DataSetInfo('/root/data/whn/nas/Database/DATABASE_UPM/chan2000/')

	Nxc = para.Nx//2 + 1

	tsteps = para.tsteps[:1]


	Fr = np.zeros(para.Ny)
	Fs = np.zeros(para.Ny)
	Fc = np.zeros(para.Ny)
	Rr = np.zeros([para.Ny, para.Nx])
	Rs = np.zeros([para.Ny, para.Nx])
	Rc = np.zeros([para.Ny, para.Nx])

	for tstep in tsteps:

		# getu = lambda j: get_layer(para.fieldpath + 'chan2000.%i.U'%tstep, j)
		# getv = lambda j: get_layer(para.fieldpath + 'chan2000.%i.V'%tstep, j)
		# getw = lambda j: get_layer(para.fieldpath + 'chan2000.%i.W'%tstep, j)

		path0 = 'rhs_full/'
		# Pressure(para).get_rhs(getu, getv, getw, path0)
		# Pressure(para).poisson(path0)


		# fluc = lambda q: q - np.mean(q, axis=(-1,-2)).reshape([-1,1,1])
		# getu = lambda j: fluc(get_layer(para.fieldpath + 'chan2000.%i.U'%tstep, j))
		# getv = lambda j: fluc(get_layer(para.fieldpath + 'chan2000.%i.V'%tstep, j))
		# getw = lambda j: fluc(get_layer(para.fieldpath + 'chan2000.%i.W'%tstep, j))

		path1 = 'rhs_slow/'
		# Pressure(para).get_rhs(getu, getv, getw, path1)
		# Pressure(para).poisson(path1)



		for j in range(para.Ny):
			print(j)

			fpr = np.fromfile(path0+'fp', dtype=np.complex64, count=para.Nz*Nxc, offset=8*para.Nz*Nxc*j).reshape([para.Nz,Nxc])
			fps = np.fromfile(path1+'fp', dtype=np.complex64, count=para.Nz*Nxc, offset=8*para.Nz*Nxc*j).reshape([para.Nz,Nxc])
			fpr -= fps

			fpr.T[0,0] = 0
			fps.T[0,0] = 0

			fpr2 = np.abs(fpr)**2
			fps2 = np.abs(fps)**2
			fprs = fpr.conj()*fps

			Fr[j] += np.sum(flipk(fpr2))
			Fs[j] += np.sum(flipk(fps2))
			Fc[j] += np.sum(flipk(fprs)).real

			Rr[j] += np.sum(ft.hfft(fpr2), axis=-2)
			Rs[j] += np.sum(ft.hfft(fps2), axis=-2)
			Rc[j] += np.sum(ft.hfft(fprs), axis=-2)


	Fr = (Fr + Fr[::-1]) / 2 / len(tsteps)
	Fs = (Fs + Fs[::-1]) / 2 / len(tsteps)
	Fc = (Fc + Fc[::-1]) / 2 / len(tsteps)
	Rr = (Rr + Rr[::-1]) / 2 / len(tsteps)
	Rs = (Rs + Rs[::-1]) / 2 / len(tsteps)
	Rc = (Rc + Rc[::-1]) / 2 / len(tsteps)

	Rr = np.roll((Rr.T/Fr).T, para.Nx//2, axis=-1)
	Rs = np.roll((Rs.T/Fs).T, para.Nx//2, axis=-1)
	Rc = np.roll((Rc.T/(Fr*Fs)**.5).T, para.Nx//2, axis=-1)

	dltxs = np.arange(-(para.Nx//2), para.Nx//2) * para.dx

	##### record #####
	fileIO.write_profiles(Fr, Fs, Fc, para, '', 'MFU', range(1, para.Ny//2+para.Ny%2))
	fileIO.write_cor1d(dltxs, Rr, Rs, Rc, para, '', 'MFU', range(1, para.Ny//2+para.Ny%2))

	##### plot ######
	import matplotlib.pyplot as plt

	yps, Fr, Fs, Fc = fileIO.read_profiles('')
	dltxps, yps, Rr, Rs, Rc = fileIO.read_cor1d('')

	fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(6.4,3.2))

	ax = axs[0]
	ax.semilogx(yps, Fr, 'r', label=r"$<p_r'p_r'>^+$")
	ax.semilogx(yps, Fs, 'b', label=r"$<p_s'p_s'>^+$")
	ax.semilogx(yps, Fc, 'g', label=r"$<p_r'p_s'>^+$")

	ax = axs[1]
	ax.plot(dltxps, Rr[np.argmin(np.abs(yps-15))], 'r', label=r'$Cor(p_r,p_r)$')
	ax.plot(dltxps, Rs[np.argmin(np.abs(yps-15))], 'b', label=r'$Cor(p_s,p_s)$')
	ax.plot(dltxps, Rc[np.argmin(np.abs(yps-15))], 'g', label=r'$Cor(p_r,p_s)$')

	ax.plot(dltxps, Rr[np.argmin(np.abs(yps-100))], 'r--')
	ax.plot(dltxps, Rs[np.argmin(np.abs(yps-100))], 'b--')
	ax.plot(dltxps, Rc[np.argmin(np.abs(yps-100))], 'g--')

	axs[0].legend()
	axs[1].legend()
	axs[0].set_xlim([1, 2000])
	axs[1].set_xlim([-500, 500])
	axs[0].set_xlabel(r'$y^+$')
	axs[1].set_xlabel(r'$\Delta x^+$')

	fig.savefig('P.png', dpi=300)
	plt.close()



# # test codes for chasing
# N = 3
# M = 5

# a = np.random.rand(N)
# b = np.random.rand(N)
# c = np.random.rand(N)
# d = np.random.rand(N)
# e = np.random.rand(N)
# f = np.random.rand(N,M) + 1j * np.random.rand(N,M)

# mat = np.diag(b)
# mat[1:,:-1] += np.diag(a[1:])
# mat[:-1,1:] += np.diag(c[:-1])
# print(np.sum(np.abs( chasing3(a,b,c,f) - np.linalg.solve(mat,f) )))

# mat = np.diag(c)
# mat[2:,:-2] += np.diag(a[2:])
# mat[1:,:-1] += np.diag(b[1:])
# mat[:-1,1:] += np.diag(d[:-1])
# mat[:-2,2:] += np.diag(e[:-2])
# print(np.sum(np.abs( chasing5(a,b,c,d,e,f) - np.linalg.solve(mat,f) )))



# a = np.random.rand(N,M)
# b = np.random.rand(N,M)
# c = np.random.rand(N,M)
# d = np.random.rand(N,M)
# e = np.random.rand(N,M)
# f = np.random.rand(N,M) + 1j * np.random.rand(N,M)

# m = 1
# mat = np.diag(c[:,m])
# mat[2:,:-2] += np.diag(a[2:,m])
# mat[1:,:-1] += np.diag(b[1:,m])
# mat[:-1,1:] += np.diag(d[:-1,m])
# mat[:-2,2:] += np.diag(e[:-2,m])
# print(np.sum(np.abs( chasing5(a,b,c,d,e,f)[:,m] - np.linalg.solve(mat,f[:,m]) )))


