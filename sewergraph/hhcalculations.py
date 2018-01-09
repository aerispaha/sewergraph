import math

def philly_storm_intensity(tc, return_period=0):
	"""
	given a tc, return the intensity of the
	Philadelphia Water Dept design storm (in/hr)
	"""
	#default, "Philly" design storm
	I = 116.0 / ( tc + 17.0)

	if return_period == 1:
		I = 100.0 / ( tc + 18.0)
	if return_period == 2:
		I = 131.0 / ( tc + 21.0)
	if return_period == 5:
		I = 171.0 / ( tc + 23.5)
	if return_period == 10:
		I = 214.0 / ( tc + 26.0)
	if return_period == 25:
		I = 252.0 / ( tc + 28.0)
	if return_period == 50:
		I = 289.0 / ( tc + 30.0)
	if return_period == 100:
		I = 325.0 / ( tc + 32.0)

	return I

def hhcalcs_on_network(G):
	"""
	For each sewer (edge) in the network, G, calculate the velocity of gravity
	flow, full-flow capacity, and full-flow travel time through the length of
	the sewer. This sets attributes in 'velocity', 'capacity',
	and 'travel_time'.

	Parameters
	----------
	G : Networkx DiGraph
		Graph of a sewer network with edges having the Slope, Height, Width
		PIPESHAPE, and Diameter parameters.
	"""
	G1 = G.copy()

	for u,v, data in G1.edges(data=True):

		#velocity
		# diameter = max( data['Diameter'], data['Height'])
		if 'slope_calculated' in data:
			slope = max(data['slope_calculated'], 0.01)
			# data['Slope'] = slope
		else:
			slope = max(data['Slope'], 0.1)
		data['slope_used_in_calcs'] = slope

		height, width = data['Height'], data['Width']
		shape, diameter = data['PIPESHAPE'], data['Diameter']

		# print height, width, shape, diameter, data['FACILITYID']
		#BUG this 'shape' should not be a string
		if 'shape' is None:
			shape = 'CIR'
			diameter = max(diameter, height)
			data['notes'] = 'assumed CIR and diam of {}'.format(diameter)


		V = mannings_velocity(diameter, slope, height, width, shape, data)
		V = max(V, 2.0) #min 2fps avoid zero div
		data['velocity'] = max(V, 2.0)

		#capacity
		A = xarea(shape, diameter, height, width)
		data['capacity'] = A * V

		#travel time
		T = (data['Shape_Leng'] / V) / 60.0 # minutes
		data['travel_time'] = T

	return G1

def slope_at_velocity(velocity, diameter, height=None, width=None, shape="CIR"):
	Rh = hydraulic_radius(shape, diameter, height, width)
	n = get_mannings(shape, diameter)
	k = (1.49 / n) * math.pow(Rh, 0.667)

	slope = (velocity / k) ** 2
	return slope

def mannings_velocity(diameter, slope, height=None, width=None, shape="CIR", data=None):

	#compute mannings velocity in full pipe
	try:
		A = xarea(shape, diameter, height, width)
		Rh = hydraulic_radius(shape, diameter, height, width)
		n = get_mannings(shape, diameter)
		V = (1.49 / n) * math.pow(Rh, 0.667) * math.pow(slope/100.0, 0.5)
	except:
		V = 2.5

	return V

def mannings_capacity(diameter, slope, height=None, width=None, shape="CIR"):

	#compute mannings flow in full pipe
	A = xarea(shape, diameter, height, width)
	Rh = hydraulic_radius(shape, diameter, height, width)
	n = get_mannings(shape, diameter)
	k = (1.49 / n) * math.pow(Rh, 0.667) * A

	Q = k * math.pow(slope/100.0, 0.5)

	return Q

def get_mannings( shape, diameter ):
	n = 0.015 #default value
	if ((shape == "CIR" or shape == "CIRCULAR") and (diameter <= 24) ):
		n = 0.015
	elif ((shape == "CIR" or shape == "CIRCULAR") and (diameter > 24) ):
		n = 0.013
	return n

def xarea( shape, diameter, height, width ):
	#calculate cross sectional area of pipe
	#supports circular, egg, and box shape
	if (shape == "CIR" or shape == "CIRCULAR"):
		return 3.1415 * (math.pow((diameter/12.0),2.0 ))/4.0
	elif (shape == "EGG" or shape == "EGG SHAPE"):
		return 0.5105* math.pow((height/12.0),2.0 )
	elif (shape == "BOX" or shape == "BOX SHAPE"):
		return height*width/144.0
	else:
		return 0

def hydraulic_radius(shape, diameter, height, width ):
	#calculate full flow hydraulic radius of pipe
	#supports circular, egg, and box shape
	if (shape == "CIR" or shape == "CIRCULAR"):
		return (diameter/12.0)/4.0
	elif (shape == "EGG" or shape == "EGG SHAPE"):
		return 0.1931* (height/12.0)
	elif (shape == "BOX" or shape == "BOX SHAPE"):
		return (height*width) / (2.0*height + 2.0*width) /12.0

def replacement_sewer_size(design_q, slope):
	"""
	return the minimum circular sewer diameter (inches) (and provided capacity)
	required to convey a given design peak flow at a set slope (ft per 100ft)
	"""
	#circular pipe sizes in inches
	sewer_diameters = [18,21,24,27,30,36,42,48,54,60,66,72,78,84]
	d = h = w = None

	for diam in sewer_diameters:
		capacity = mannings_capacity(diam, slope=slope, shape="CIR")
		if capacity > design_q:
			d = diam
			return d, h, w, capacity

	#if we haven't met the design_q with a circulare pipe,
	#iterate on rectangular BOX sewers
	h = 48
	w = 66
	while capacity < design_q:
		capacity = mannings_capacity(height=h, width=w, slope=slope,
									shape='BOX', diameter=None)
		h += 6
		w += 6

	return d, h, w, capacity
