import numpy as np


##### write #####

def write_HL(path, kxps, yps, HL):
	header = \
		'Title = "Linear transfer kernel"\n' + \
		'variables = "%s", "%s", "%s", "%s", "%s", "%s", "%s"\n' \
		% (	'k<sub>x</sub><sup>+</sup>',
			'y<sup>+</sup>',
			'real(H<sub>L</sub>)',
			'imag(H<sub>L</sub>)',
			'|H<sub>L</sub>|',
			'<greek>f</greek>',
			'<greek>l</greek><sub>x</sub><sup>+</sup>' ) + \
		'zone t = "%s", i = %i, j = %i' %('DNS4200', len(kxps), len(yps))

	data = np.empty([7, len(yps), len(kxps)])
	for j, yp in enumerate(yps):
		for i, kx in enumerate(kxps):
			data[:,j,i] = [
				kx, yp,
				HL[j,i].real,
				HL[j,i].imag,
				np.abs(HL[j,i]),
				np.angle(HL[j,i]),
				2*np.pi/(kx if kx else 1e-10), ]

	data = np.transpose([col.ravel() for col in data])

	pame = path + 'HL.dat'

	np.savetxt(pame, data, header=header, comments='')

	return pame

def write_Alpha(path, yps, Alpha):
	header = \
		'Title = "Superposition coefficient"\n' + \
		'variables = "%s", "%s", "%s", "%s"\n' \
		% (	'y<sup>+</sup>',
			'<greek>A</greek>',
			'<greek>D</greek>x <sub><greek>G</greek></sub> <sup>+</sup>',
			'<greek>D</greek>z <sub><greek>G</greek></sub> <sup>+</sup>', ) + \
		'zone t = "%s", i = %i' %('DNS4200', len(yps))

	data = np.transpose([yps] + list(Alpha.T))

	pame = path + 'Alpha.dat'

	np.savetxt(pame, data, header=header, comments='')

	return pame

def write_Gamma(path, yps, Gamma):
	header = \
		'Title = "Modulation coefficient"\n' + \
		'variables = "%s", "%s", "%s", "%s"\n' \
		% (	'y<sup>+</sup>',
			'<greek>G</greek>',
			'<greek>D</greek>x <sub><greek>G</greek></sub> <sup>+</sup>',
			'<greek>D</greek>z <sub><greek>G</greek></sub> <sup>+</sup>', ) + \
		'zone t = "%s", i = %i' %('DNS4200', len(yps))

	data = np.transpose([yps] + list(Gamma.T))

	pame = path + 'Gamma.dat'

	np.savetxt(pame, data, header=header, comments='')

	return pame

def write_AM(path, yp, dltxps, dltzps, AM):
	header = \
		'Title = "Amplitude modulation intensity"\n' + \
		'variables = "%s", "%s", "%s"\n' \
		% (	'<greek>D</greek>x<sup>+</sup>',
			'<greek>D</greek>z<sup>+</sup>',
			'AM' ) + \
		'zone t = "%s", i = %i, j = %i' %('yp%.2f'%yp, len(dltxps), len(dltzps))

	data = np.empty([3, len(dltzps), len(dltxps)])
	for j, dltzp in enumerate(dltzps):
		for i, dltxp in enumerate(dltxps):
			data[:,j,i] = [ dltxp, dltzp, AM[j,i] ]

	data = np.reshape(data, [len(data),-1]).T

	pame = path + 'AM_yp%.2f.dat'%yp

	np.savetxt(pame, data, header=header, comments='')

	return pame


##### read #####

def read_HL(path):
	data = np.loadtxt(path + 'HL.dat', skiprows=3)

	nx, ny = 0, 0
	with open(path + 'HL.dat') as fp:
		for term in fp.readlines()[2].split(','):
			if term.strip()[0] == 'i': nx = int(term.split('=')[-1])
			if term.strip()[0] == 'j': ny = int(term.split('=')[-1])

	kxps = data[:nx,0]
	yps = data[::nx,1]
	HL = np.reshape(data[:,2] + 1j * data[:,3], [ny, nx])

	return kxps, yps, HL

def read_Alpha(path):
	data = np.loadtxt(path + 'Alpha.dat', skiprows=3)
	return data[:,0], data[:,1:] # yps, alf, dltxp, dltyp


def read_Gamma(path):
	data = np.loadtxt(path + 'Gamma.dat', skiprows=3)
	return data[:,0], data[:,1:] # yps, gma, dltxps, dltzps








