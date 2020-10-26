#!/root/Software/anaconda3/bin/python3
import numpy as np
from numpy.fft import hfft,ihfft
from os import listdir, system

import basic
from basic import *
from plot import *



class PDF:
	def __init__(self):
		pass

	@staticmethod
	def get_samples(get_layer_fluc_plus, yp, tsteps=None):

		j = np.argmin(np.abs(yps-yp))

		uset, vset, wset = [], [], []

		for t in (tsteps if tsteps else basic.tsteps):
			uset.append( get_layer_fluc_plus('chan2000.%i.U'%t, j) )
			vset.append( get_layer_fluc_plus('chan2000.%i.V'%t, j) )
			wset.append( get_layer_fluc_plus('chan2000.%i.W'%t, j) )
			uset.append( get_layer_fluc_plus('chan2000.%i.U'%t, Ny-j-1) )
			vset.append(-get_layer_fluc_plus('chan2000.%i.V'%t, Ny-j-1) )
			wset.append( get_layer_fluc_plus('chan2000.%i.W'%t, Ny-j-1) )

		return	np.ravel(uset), \
				np.ravel(vset), \
				np.ravel(wset)

	@staticmethod
	def calc_jpdf(uset, vset, n1=100, n2=100):

		u0, v0 = np.min(uset), np.min(vset)
		u1, v1 = np.max(uset), np.max(vset)

		us = np.linspace(u0, u1, n1)
		vs = np.linspace(v0, v1, n2)

		du = us[1] - us[0]
		dv = vs[1] - vs[0]

		pdf = np.zeros([len(vs), len(us)])

		idx = np.vstack([(uset-u0)/du, (vset-v0)/dv]).astype(int)
		idx, cnt = np.unique(idx, return_counts=True, axis=-1)

		pdf[idx[1], idx[0]] += cnt / len(uset) / (du * dv)

		return us, vs, pdf

	@staticmethod
	def write_jpdf(pame, us, vs, pdf, casename='jpdf'):
		header = \
			'Title = "Joint PDF of u and v"\n' + \
			'variables = "%s", "%s", "%s"\n' \
			% (	'u<sup>+</sup>',
				'v<sup>+</sup>',
				'pdf' ) + \
			'zone t = "%s", i = %i, j = %i' %(casename, len(us), len(vs))

		data = np.empty([3, len(vs), len(us)])

		for j, v in enumerate(vs):
			for i, u in enumerate(us):
				data[:,j,i] = u, v, pdf[j,i]

		data = np.transpose([col.ravel() for col in data])

		np.savetxt(pame, data, header=header, comments='')

		return pame

	@staticmethod
	def read_jpdf(pame):
		data = np.loadtxt(pame, skiprows=3)

		nx, ny = 0, 0
		with open(pame) as fp:
			for term in fp.readlines()[2].split(','):
				if term.strip()[0] == 'i': nx = int(term.split('=')[-1])
				if term.strip()[0] == 'j': ny = int(term.split('=')[-1])

		us = data[:nx,0]
		vs = data[::nx,1]
		pdf = np.reshape(data[:,2], [ny,nx])

		return us, vs, pdf



if __name__ == '__main__':

	datapath = '../../../HoyasJimenez2008/fields/'
	workpath = 'results/'

	get_fluc = lambda q: q - np.mean(q)
	get_layer_fluc_plus = lambda fn, j: get_fluc(get_layer(datapath+fn, j)) / uc

	uset, vset, wset = PDF.get_samples(get_layer_fluc_plus, 15)

	us, vs, pdf_uv = PDF.calc_jpdf(uset, vset)
	us, ws, pdf_uw = PDF.calc_jpdf(uset, wset)

	PDF.write_jpdf(workpath + 'jpdf_uv.dat', us, vs, pdf_uv, casename='DNS2000')
	PDF.write_jpdf(workpath + 'jpdf_uw.dat', us, ws, pdf_uw, casename='DNS2000')

	plot_jpdf(workpath + 'jpdf_uv.png', us, vs, pdf_uv)
	plot_jpdf(workpath + 'jpdf_uw.png', us, ws, pdf_uw)





