import numpy as np
import os


class Write:
	def __init__(self, para, path, casename='test'):
		self.para = para
		self.workpath = path
		self.casename = casename

	def write_prof(self, stas, jrange):

		para = self.para

		header = \
			'Title = "profiles of basic statistics"\n' + \
			'variables = "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s"\n' \
			% (	"y<sup>+</sup>",
				"<u><sup>+</sup>", "<v><sup>+</sup>", "<w><sup>+</sup>", "<p><sup>+</sup>", \
				"<u'u'><sup>+</sup>", "<v'v'><sup>+</sup>", "<w'w'><sup>+</sup>", \
				"<u'v'><sup>+</sup>", "<v'w'><sup>+</sup>", "<u'w'><sup>+</sup>", ) + \
			'zone t = "%s", i = %i' %(self.casename, len(jrange))

		um, vm, wm = map(lambda a: a/para.uc,    (stas.Um,  stas.Vm,  stas.Wm))
		uu, vv, ww = map(lambda a: a/para.uc**2, (stas.R11, stas.R22, stas.R33))
		uv, vw, uw = map(lambda a: a/para.uc**2, (stas.R12, stas.R23, stas.R13))
		pm = np.zeros(para.Ny)

		data = np.transpose([para.yps, um, vm, wm, pm, uu, vv, ww, uv, vw, uw])[jrange]

		np.savetxt(self.workpath+"profiles.dat", data, header=header, comments='')

	def write_es1d_xy(self, stas, irange, jrange):

		para = self.para

		header = \
			'Title = "1D streamwise energy spectra"\n' + \
			'variables = "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s"\n' \
			% (	"log<sub>10</sub>(<greek>l</greek><sub>x</sub><sup>+</sup>)",
				"log<sub>10</sub>(y<sup>+</sup>)",
				"k<sub>x</sub>E<sub>uu</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>x</sub>E<sub>vv</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>x</sub>E<sub>ww</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>x</sub>E<sub>pp</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>x</sub>E<sub>uv</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>x</sub>E<sub>vw</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>x</sub>E<sub>uw</sub>/u<sub><greek>t</greek></sub><sup>2</sup>"	) + \
			'zone t = "%s", i = %i, j = %i' %(self.casename, len(jrange), len(irange))

		data = np.empty([9, len(irange), len(jrange)])
		for ii,i in enumerate(irange):
			for jj,j in enumerate(jrange):
				data[:, ii, jj] = [
					para.kxs[i],
					para.ys[j],
					np.sum(stas.Euu[j,:,i]),
					np.sum(stas.Evv[j,:,i]),
					np.sum(stas.Eww[j,:,i]),
					0, # Epp = 0
					np.sum(stas.Euv[j,:,i]),
					np.sum(stas.Evw[j,:,i]),
					np.sum(stas.Euw[j,:,i])	]

		data[2:] *= data[0] / (2*np.pi / para.Lx) / para.uc**2
		data[5] *= para.uc**2 / para.pc**2
		data[0] = 2*np.pi / data[0]
		data[:2] = np.log10(data[:2] / para.lc)
		data = np.array([np.ravel(temp) for temp in data]).T

		pame = self.workpath + "ES1D_xy.dat"
		np.savetxt(pame, data, header=header, comments='')
		if not os.system("preplot " + pame):
			pass
			# os.system("rm -f " + pame)

	def write_es1d_zy(self, stas, krange, jrange):

		para = self.para

		header = \
			'Title = "1D spanwise energy spectra"\n' + \
			'variables = "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s"\n' \
			% (	"log<sub>10</sub>(<greek>l</greek><sub>z</sub><sup>+</sup>)",
				"log<sub>10</sub>(y<sup>+</sup>)",
				"k<sub>z</sub>E<sub>uu</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>z</sub>E<sub>vv</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>z</sub>E<sub>ww</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>z</sub>E<sub>pp</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>z</sub>E<sub>uv</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>z</sub>E<sub>vw</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>z</sub>E<sub>uw</sub>/u<sub><greek>t</greek></sub><sup>2</sup>"	) + \
			'zone t = "%s", i = %i, j = %i' %(self.casename, len(jrange), len(krange))

		data = np.empty([9, len(krange), len(jrange)])
		for kk,k in enumerate(krange):
			for jj,j in enumerate(jrange):
				data[:, kk, jj] = [
					para.kzs[k],
					para.ys[j],
					np.sum(stas.Euu[j,k]),
					np.sum(stas.Evv[j,k]),
					np.sum(stas.Eww[j,k]),
					0,
					np.sum(stas.Euv[j,k]),
					np.sum(stas.Evw[j,k]),
					np.sum(stas.Euw[j,k])	]

		data[2:] *= data[0] / (2*np.pi / para.Lz) / para.uc**2
		data[5] *= para.uc**2 / para.pc**2
		data[0] = 2*np.pi / data[0]
		data[:2] = np.log10(data[:2] / para.lc)
		data = np.array([np.ravel(temp) for temp in data]).T

		pame = self.workpath + "ES1D_zy.dat"
		np.savetxt(pame, data, header=header, comments='')
		if not os.system("preplot " + pame):
			pass
			# os.system("rm -f " + pame)


	@staticmethod
	def write_es2d(filename, para, casename, Euu, Evv, Eww, Euv):

		irange = range(1, para.Nx//2+1)
		krange = range(1, para.Nz//2+1)

		header = \
			'Title = "2D energy spectra"\n' + \
			'variables = "%s", "%s", "%s", "%s", "%s", "%s"\n' \
			% (	"log<sub>10</sub>(<greek>l</greek><sub>x</sub><sup>+</sup>)",
				"log<sub>10</sub>(<greek>l</greek><sub>z</sub><sup>+</sup>)",
				"k<sub>x</sub>k<sub>z</sub>E<sub>uu</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>x</sub>k<sub>z</sub>E<sub>vv</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>x</sub>k<sub>z</sub>E<sub>ww</sub>/u<sub><greek>t</greek></sub><sup>2</sup>",
				"k<sub>x</sub>k<sub>z</sub>E<sub>uv</sub>/u<sub><greek>t</greek></sub><sup>2</sup>", ) + \
			'zone t = "%s", i = %i, j = %i' %(casename, len(irange), len(krange))

		data = np.empty([6, len(krange), len(irange)])

		# # loop implementation is too slow
		# for kk,k in enumerate(krange):
		# 	for ii,i in enumerate(irange):
		# 		data[:, kk, ii] = [
		# 			np.log10(2*np.pi / para.kxs[i] / para.lc),
		# 			np.log10(2*np.pi / para.kzs[k] / para.lc),
		# 			Euu[k,i] * para.kxs[i] * para.kzs[k] / (4*np.pi**2 / para.Lx / para.Lz) / para.uc**2,
		# 			Evv[k,i] * para.kxs[i] * para.kzs[k] / (4*np.pi**2 / para.Lx / para.Lz) / para.uc**2,
		# 			Eww[k,i] * para.kxs[i] * para.kzs[k] / (4*np.pi**2 / para.Lx / para.Lz) / para.uc**2,
		# 			Euv[k,i] * para.kxs[i] * para.kzs[k] / (4*np.pi**2 / para.Lx / para.Lz) / para.uc**2,
		# 			]

		data[0]      = np.log10(2*np.pi/para.lc / para.kxs[irange])
		data[1].T[:] = np.log10(2*np.pi/para.lc / para.kzs[krange])
		data[2] = (Euu[krange].T * para.kzs[krange])[irange].T * para.kxs[irange] / (4*np.pi**2/para.Lx/para.Lz) / para.uc**2
		data[3] = (Evv[krange].T * para.kzs[krange])[irange].T * para.kxs[irange] / (4*np.pi**2/para.Lx/para.Lz) / para.uc**2
		data[4] = (Eww[krange].T * para.kzs[krange])[irange].T * para.kxs[irange] / (4*np.pi**2/para.Lx/para.Lz) / para.uc**2
		data[5] = (Euv[krange].T * para.kzs[krange])[irange].T * para.kxs[irange] / (4*np.pi**2/para.Lx/para.Lz) / para.uc**2

		data = np.transpose([_.ravel() for _ in data])

		np.savetxt(filename, data, fmt='%.8e', header=header, comments='')
		if not os.system("preplot " + filename):
			pass
			# os.system("rm -f " + filename)





