import numpy as np
import fileIO



def calc_jpdf(uset_, vset_, n1=100, n2=100):

	uset = np.ravel(uset_)
	vset = np.ravel(vset_)

	if len(uset) != len(vset):
		print('Samples do not align!')
		exit()

	u0, v0 = np.min(uset), np.min(vset)
	u1, v1 = np.max(uset), np.max(vset)

	us = np.linspace(u0, u1, n1)
	vs = np.linspace(v0, v1, n2)

	du = us[1] - us[0]
	dv = vs[1] - vs[0]

	pdf = np.zeros([len(vs), len(us)])

	idx = np.vstack([(uset-u0)/du+.5, (vset-v0)/dv+.5]).astype(int)
	idx, cnt = np.unique(idx, return_counts=True, axis=-1)

	pdf[idx[1], idx[0]] += cnt / len(uset) / (du * dv)

	return us, vs, pdf


def get_samples(para, yp, tskip=1):

	get_fluc_plus = lambda q: (q - np.mean(q)) / para.uc
	tsteps = para.tsteps[::tskip]
	ny = para.Ny-1

	j = np.argmin(np.abs(para.yps-yp))

	uset, vset, wset = [], [], []

	for t in tsteps:
		print('reading step %i'%t)
		for jj in [j, ny-j]:
			uset.append( get_fluc_plus(fileIO.get_layer(para.fieldpath+'chan2000.%i.U'%t, jj)) )
			vset.append( get_fluc_plus(fileIO.get_layer(para.fieldpath+'chan2000.%i.V'%t, jj)) * (1 if jj==j else -1) )
			wset.append( get_fluc_plus(fileIO.get_layer(para.fieldpath+'chan2000.%i.W'%t, jj)) )

	return [np.ravel(_) for _ in (uset, vset, wset)]

def get_samples_filt(para, yp, dx):

	from filtdns import FDNS_CUDA as FDNS
	from tools import Tools_cuda as tool

	fdns = FDNS(para)
	ny = para.Ny-1

	jj = np.argmin(np.abs(para.yps - yp))

	uset, vset, wset = [], [], []

	for t in para.tsteps:
		print('reading step %i'%t)

		getlyr_u = lambda j: fileIO.get_layer(para.fieldpath + 'chan2000.%d.U'%t, j) / para.uc
		getlyr_v = lambda j: fileIO.get_layer(para.fieldpath + 'chan2000.%d.V'%t, j) / para.uc
		getlyr_w = lambda j: fileIO.get_layer(para.fieldpath + 'chan2000.%d.W'%t, j) / para.uc

		uf, um = fdns.filt(getlyr_u, dx, dx, dx)
		vf, vm = fdns.filt(getlyr_v, dx, dx, dx)
		wf, wm = fdns.filt(getlyr_w, dx, dx, dx)

		uset.append( tool.phys(uf[[jj, ny-jj]]) )
		vset.append( tool.phys(vf[[jj, ny-jj]]) * np.reshape([1,-1], [-1,1,1]) )
		wset.append( tool.phys(wf[[jj, ny-jj]]) )

	return [np.ravel(_) for _ in (uset, vset, wset)]


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


if __name__ == '__main__':
	import basic2k as basic

	para = basic.DataSetInfo('/home/whn/nasdata/chan2000/')

	yp = 174

	# uset, vset, wset = get_samples(para, yp, tskip=1)
	uset, vset, wset = get_samples_filt(para, yp, dx=100/2000)

	us, vs, pdf_uv = calc_jpdf(uset, vset)
	us, ws, pdf_uw = calc_jpdf(uset, wset)

	write_jpdf('jpdf_uv.dat', us, vs, pdf_uv, casename='DNS2000')
	write_jpdf('jpdf_uw.dat', us, ws, pdf_uw, casename='DNS2000')



