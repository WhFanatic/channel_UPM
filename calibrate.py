#!/root/Software/anaconda3/bin/python3
import numpy as np
import torch as tc
import numpy.fft as ft
from basic2k import get_layer
from tools import Tools_cuda as tool


class Calibrate:

	def __init__(self, para, yop):
		self.para = para

		self.jo = np.argmin(np.abs(para.yps - yop))
		self.js = [j for j in range(para.Ny) if 5<para.yps[j]<105]
		self.yps = para.yps[self.js]

		self.kxps = para.kxs * para.lc
		self.kzps = ft.fftfreq(para.Nz, d=para.dz/(2*np.pi)) * para.lc

		self.dltxps = para.dx/para.lc * np.arange(-para.Nx//2, para.Nx//2)
		self.dltzps = para.dz/para.lc * np.arange(-para.Nz//2, para.Nz//2)

		lxpc = 1e3 * np.pi # streamwise filter size in inner scale
		lzpc = 1e2 * np.pi # spanwise filter size in inner scale

		# select out modes that are larger than MFUs in both directions
		self.mfuter = (self.kxps < 2*np.pi/lxpc) * np.reshape(np.abs(self.kzps) < 2*np.pi/lzpc, (-1,1))

		# PIO parameters that need to be calibrated
		self.HL = np.zeros([len(self.yps), len(self.kxps)], dtype=complex)

		self.Alpha_v = np.zeros([len(self.yps), 3])
		self.Alpha_w = np.zeros([len(self.yps), 3])
	
		self.Gamma_up = np.zeros([len(self.yps), 3])
		self.Gamma_um = np.zeros([len(self.yps), 3])
		self.Gamma_v = np.zeros([len(self.yps), 3])
		self.Gamma_w = np.zeros([len(self.yps), 3])

	def super(self):
		para = self.para
		Nx = para.Nx
		Nz = para.Nz

		ts1 = para.tsteps1
		ts2 = para.tsteps2
		ts3 = para.tsteps3

		# outer signals
		fuO  = ft.ihfft (self.__getu(ts1, self.jo))
		fvOL = tool.spec(self.__getv(ts2, self.jo)) * self.mfuter
		fwOL = tool.spec(self.__getw(ts3, self.jo)) * self.mfuter

		# linear transfer kernel for U
		fuOc = np.conj(fuO)

		for j, H in zip(self.js, self.HL):
			print('Sup U:', j)
			H[:] = np.mean(fuOc * ft.ihfft(self.__getu(ts1, j)), axis=(0,1))

		self.HL /= np.mean(np.abs(fuO)**2, axis=(0,1))

		self.smooth_HL(self.kxps, self.HL)

		# superposition shift & coefficient for V & W
		var2v = np.var(tool.phys(fvOL))
		var2w = np.var(tool.phys(fwOL))

		for jj, j in enumerate(self.js):
			print('Sup VW:', j)

			fvL = tool.spec(self.__getv(ts2, j)) * self.mfuter
			fwL = tool.spec(self.__getw(ts3, j)) * self.mfuter

			var1v = np.var(tool.phys(fvL))
			var1w = np.var(tool.phys(fwL))

			cor2 = tool.roll2(tool.phys(np.conj(fvL) * fvOL), Nx//2, Nz//2)
			cor3 = tool.roll2(tool.phys(np.conj(fwL) * fwOL), Nx//2, Nz//2)

			arg2 = np.array([np.argmax(cor[Nz//2:]+cor[Nz//2:0:-1]) for cor in cor2])
			arg3 = np.array([np.argmax(cor[Nz//2:]+cor[Nz//2:0:-1]) for cor in cor3])

			dltxp2, dltzp2 = np.mean(self.dltxps[arg2%Nx]), np.mean(self.dltzps[Nz//2+arg2//Nx])
			dltxp3, dltzp3 = np.mean(self.dltxps[arg3%Nx]), np.mean(self.dltzps[Nz//2+arg3//Nx])

			self.Alpha_v[jj] = (min(self.yps[jj]/13.5,1)*var1v/var2v)**.5, dltxp2, dltzp2
			self.Alpha_w[jj] = (min(self.yps[jj]/13.5,1)*var1w/var2w)**.5, dltxp3, dltzp3

	def modul(self):
		para = self.para
		lc = para.lc
		dx = para.dx
		dz = para.dz

		ts1 = para.tsteps1
		ts2 = [t for t in para.tsteps1 if t in para.tsteps2]
		ts3 = [t for t in para.tsteps1 if t in para.tsteps3]

		ts12 = np.transpose([(i,i+len(ts1)) for i,t in enumerate(ts1) if t in ts2]).ravel()
		ts13 = np.transpose([(i,i+len(ts1)) for i,t in enumerate(ts1) if t in ts3]).ravel()

		fuO = ft.ihfft(self.__getu(ts1, self.jo))

		for jj, j in enumerate(self.js):
			print('Mod U:', j)

			H = self.HL[jj]
			I = tool.normalize(np.abs(H))

			# modulating signal
			uL = ft.hfft(H * fuO)

			# small-scale fluctuations
			uS = self.__getu(ts1, j) - uL

			##### when CUDA is available #####
			I  = tc.tensor(I,  dtype=tc.float, device='cuda')
			uL = tc.tensor(uL, dtype=tc.float, device='cuda')
			uS = tc.tensor(uS, dtype=tc.float, device='cuda')

			# modulation shift
			dltxp1, dltzp1 = self.cuda_shift(I, uS, uL, tool.envelup)
			dltxp2, dltzp2 = self.cuda_shift(I, uS, uL, tool.envelow)

			# modulation intensity
			i1, k1 = -int(dltxp1*lc/dx), -int(dltzp1*lc/dz)
			i2, k2 = -int(dltxp2*lc/dx), -int(dltzp2*lc/dz)

			AM1 = lambda g: self.cuda_AM(g, I, uS, uL, i1, k1, tool.envelup)
			AM2 = lambda g: self.cuda_AM(g, I, uS, uL, i2, k2, tool.envelow)

			gma1 = tool.newton(AM1, .06)
			gma2 = tool.newton(AM2, .06)
			########################################

			self.Gamma_up[jj] = gma1, dltxp1, dltzp1
			self.Gamma_um[jj] = gma2, dltxp2, dltzp2

		del uL, uS # to save GPU memory

		for jj, j in enumerate(self.js):
			print('Mod VW:', j)

			H = self.HL[jj]
			I = tool.normalize(np.abs(H))

			# modulating signal
			uL = ft.hfft(H * fuO)

			# small-scale fluctuations
			vS = tool.phys(tool.spec(self.__getv(ts2, j)) * (1-self.mfuter))
			wS = tool.phys(tool.spec(self.__getw(ts3, j)) * (1-self.mfuter))

			##### when CUDA is available #####
			I  = tc.tensor(I,  dtype=tc.float, device='cuda')
			uL = tc.tensor(uL, dtype=tc.float, device='cuda')
			vS = tc.tensor(vS, dtype=tc.float, device='cuda')
			wS = tc.tensor(wS, dtype=tc.float, device='cuda')

			# modulation shift
			dltxp2, dltzp2 = self.cuda_shift(I, vS, uL[ts12], tool.envelop)
			dltxp3, dltzp3 = self.cuda_shift(I, wS, uL[ts13], tool.envelop)

			# modulation intensity
			i2, k2 = -int(dltxp2*lc/dx), -int(dltzp2*lc/dz)
			i3, k3 = -int(dltxp3*lc/dx), -int(dltzp3*lc/dz)

			AM2 = lambda g: self.cuda_AM(g, I, vS, uL[ts12], i2, k2, tool.envelop)
			AM3 = lambda g: self.cuda_AM(g, I, wS, uL[ts13], i3, k3, tool.envelop)

			gma2 = tool.newton(AM2, .06)
			gma3 = tool.newton(AM3, .06)
			########################################

			self.Gamma_v[jj] = gma2, dltxp2, dltzp2
			self.Gamma_w[jj] = gma3, dltxp3, dltzp3


	def cuda_shift(self, I, qS_, uL_, get_env):
		''' compute the modulation shift on CUDA '''
		Nx = self.para.Nx
		Nz = self.para.Nz

		dltxp = 0
		dltzp = 0

		for qS, uL in zip(qS_, uL_):

			e = get_env(qS)
			fe = tc.rfft(e, signal_ndim=1)
			fe.T[0].T[:] *= I
			fe.T[1].T[:] *= I
			eL = tc.irfft(fe, signal_ndim=1, signal_sizes=e.shape[-1:])

			cor = tc.roll(tool.corr2p(eL, uL), (Nx//2,Nz//2), (-1,-2))

			arg = tc.argmax(cor[Nz//2:] + tc.flip(cor[1:Nz//2+1], (0,)))

			dltxp += self.dltxps[arg%Nx]
			dltzp += self.dltzps[Nz//2+arg//Nx]

		return dltxp/len(qS_), dltzp/len(qS_)

	def cuda_AM(self, gma, I, qS_, uL_, i, k, get_env):
		''' compute the modulation intensity on CUDA '''
		cor = 0
		for qS, uL in zip(qS_, uL_): # compute layer by layer due to limited memory on GPU

			e = get_env(qS / (self.para.uc + gma * tc.roll(uL, (i,k), (-1,-2))))
			fe = tc.rfft(e, signal_ndim=1)
			fe.T[0].T[:] *= I
			fe.T[1].T[:] *= I
			eL = tc.irfft(fe, signal_ndim=1, signal_sizes=e.shape[-1:])
			cor += tool.corr(eL, uL)

		return cor / len(qS_)

	@staticmethod
	def smooth_HL(kxp, HL):
		for H in HL:
			# 3-point filter
			H[:] = np.mean([H, H,
				np.hstack([H[0], H[:-1]]),
				np.hstack([H[1:], H[-1]])], axis=0)
			# roll out the uncorrelated small scales
			thres = np.argmax(np.angle(H) > 0)
			H[thres:] *= 1 - np.cos(.5*np.pi * kxp[thres] / kxp[thres:])


	def read_super(self, path):
		_, _, self.HL = read_HL(path + 'U/')
		_, self.Alpha_v = read_Alpha(path + 'V/')
		_, self.Alpha_w = read_Alpha(path + 'W/')

	def __getu(self, ts, j):
		us1 = np.array([get_layer(self.para.fieldpath + 'chan2000.%i.U'%t, j) for t in ts])
		us2 = np.array([get_layer(self.para.fieldpath + 'chan2000.%i.U'%t, self.para.Ny-1-j) for t in ts])
		return np.concatenate([us1 - np.mean(us1), us2 - np.mean(us2)])

	def __getv(self, ts, j):
		vs1 = np.array([get_layer(self.para.fieldpath + 'chan2000.%i.V'%t, j) for t in ts])
		vs2 = np.array([get_layer(self.para.fieldpath + 'chan2000.%i.V'%t, self.para.Ny-1-j) for t in ts])
		return np.concatenate([vs1 - np.mean(vs1), -vs2 + np.mean(vs2)])

	def __getw(self, ts, j):
		ws1 = np.array([get_layer(self.para.fieldpath + 'chan2000.%i.W'%t, j) for t in ts])
		ws2 = np.array([get_layer(self.para.fieldpath + 'chan2000.%i.W'%t, self.para.Ny-1-j) for t in ts])
		return np.concatenate([ws1 - np.mean(ws1), ws2 - np.mean(ws2)])



if __name__ == '__main__':

	datapath = '/root/data/whn/nas/Database/DATABASE_UPM/chan4200/'
	workpath = 'results/'

	para = DataSetInfo(datapath)
	calib = Calibrate(para, 3.9*para.Ret**.5)

	calib.super()

	write_HL   (workpath + 'U/', calib.kxps, calib.yps, calib.HL)
	write_Alpha(workpath + 'V/', calib.yps, calib.Alpha_v)
	write_Alpha(workpath + 'W/', calib.yps, calib.Alpha_w)

	calib.modul()

	write_Gamma(workpath + 'U/UP/', calib.yps, calib.Gamma_up)
	write_Gamma(workpath + 'U/UM/', calib.yps, calib.Gamma_um)
	write_Gamma(workpath + 'V/',    calib.yps, calib.Gamma_v)
	write_Gamma(workpath + 'W/',    calib.yps, calib.Gamma_w)




## when cuda not available

		# # modulation shift
		# upEL = hfft(I * ihfft(envelup(uS)))
		# umEL = hfft(I * ihfft(envelow(uS)))

		# cor1 = roll2(corr2p(upEL, uL), Nx//2, Nz//2)
		# cor2 = roll2(corr2p(umEL, uL), Nx//2, Nz//2)

		# arg1 = np.array([np.argmax(cor[Nz//2:]+cor[Nz//2:0:-1]) for cor in cor1])
		# arg2 = np.array([np.argmax(cor[Nz//2:]+cor[Nz//2:0:-1]) for cor in cor2])

		# dltxp1, dltzp1 = np.mean(dltxps[arg1%Nx]), np.mean(dltzps[Nz//2+arg1//Nx])
		# dltxp2, dltzp2 = np.mean(dltxps[arg2%Nx]), np.mean(dltzps[Nz//2+arg2//Nx])

		# # modulation intensity
		# i1, k1 = -int(dltxp1*lc/dx), -int(dltzp1*lc/dz)
		# i2, k2 = -int(dltxp2*lc/dx), -int(dltzp2*lc/dz)

		# uL1 = roll2(uL, i1, k1)
		# uL2 = roll2(uL, i2, k2)

		# AM1 = lambda g: corr(hfft(I * ihfft(envelup(uS/(uc+g*uL1)))), uL)
		# AM2 = lambda g: corr(hfft(I * ihfft(envelow(uS/(uc+g*uL2)))), uL)

		# gma1 = newton(AM1, .06, maxiter=maxiter)
		# gma2 = newton(AM2, .06, maxiter=maxiter)




		# # modulation shift
		# vEL = hfft(I * ihfft(envelop(vS)))
		# wEL = hfft(I * ihfft(envelop(wS)))

		# cor2 = roll2(corr2p(vEL, uL[ts12]), Nx//2, Nz//2)
		# cor3 = roll2(corr2p(wEL, uL[ts13]), Nx//2, Nz//2)

		# arg2 = np.array([np.argmax(cor[Nz//2:]+cor[Nz//2:0:-1]) for cor in cor2])
		# arg3 = np.array([np.argmax(cor[Nz//2:]+cor[Nz//2:0:-1]) for cor in cor3])

		# dltxp2, dltzp2 = np.mean(dltxps[arg2%Nx]), np.mean(dltzps[Nz//2+arg2//Nx])
		# dltxp3, dltzp3 = np.mean(dltxps[arg3%Nx]), np.mean(dltzps[Nz//2+arg3//Nx])

		# # modulation intensity
		# i2, k2 = -int(dltxp2*lc/dx), -int(dltzp2*lc/dz)
		# i3, k3 = -int(dltxp3*lc/dx), -int(dltzp3*lc/dz)

		# uL2 = roll2(uL[ts12], i2, k2)
		# uL3 = roll2(uL[ts13], i3, k3)
		
		# AM2 = lambda g: corr(hfft(I * ihfft(envelop(vS/(uc+g*uL2)))), uL[ts12])
		# AM3 = lambda g: corr(hfft(I * ihfft(envelop(wS/(uc+g*uL3)))), uL[ts13])

		# gma2 = newton(AM2, .06, maxiter=maxiter)
		# gma3 = newton(AM3, .06, maxiter=maxiter)







