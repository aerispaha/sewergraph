import math

#define default hydraulic params
default_min_slope = 0.01 # percent - assumed when slope is null
default_TC_slope = 5.0 # percent - conservatively assumed for travel time calculation when slope
pipeSizesAvailable = [18,21,24,27,30,36,42,48,54,60,66,72,78,84] #circular pipe sizes in inches

def philly_storm_intensity(tc):
	"""
	given a tc, return the intensity of the
	Philadelphia Water Dept design storm (in/hr)
	"""
	I = 116.0 / ( tc + 17.0)
	return I

def hhcalcs_on_network(G):

	G1 = G.copy()

	for u,v, data in G1.edges_iter(data=True):

		#velocity
		# diameter = max( data['Diameter'], data['Height'])
		if 'calculated_slope' in data:
			slope = max(data['calculated_slope'], 0.2)
			data['Slope'] = slope
		else:
			slope = max(data['Slope'], 0.2)
		data['slope_used_in_calcs'] = slope

		height, width = data['Height'], data['Width']
		shape, diameter = data['PIPESHAPE'], data['Diameter']

		# print height, width, shape, diameter, data['FACILITYID']
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

def mannings_velocity(diameter, slope, height=None, width=None, shape="CIR", data=None):

	#compute mannings velocity in full pipe
	try:
		A = xarea(shape, diameter, height, width)
		Rh = hydraulicRadius(shape, diameter, height, width)
		n = getMannings(shape, diameter)
		V = (1.49 / n) * math.pow(Rh, 0.667) * math.pow(slope/100.0, 0.5)
	except:
		print 'cannot compute, assume 2fps, {}'.format(data['FACILITYID'])
		V = 2.0

	return V

def getMannings( shape, diameter ):
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

def  minSlope( slope ):
	#replaces null slope value with the assumed minimum 0.01%
	if slope == None:
		return 0.01
	else:
		return slope

def hydraulicRadius(shape, diameter, height, width ):
	#calculate full flow hydraulic radius of pipe
	#supports circular, egg, and box shape
	if (shape == "CIR" or shape == "CIRCULAR"):
		return (diameter/12.0)/4.0
	elif (shape == "EGG" or shape == "EGG SHAPE"):
		return 0.1931* (height/12.0)
	elif (shape == "BOX" or shape == "BOX SHAPE"):
		return (height*width) / (2.0*height + 2.0*width) /12.0


def manningsCapacity(diameter, slope, height=None, width=None, shape="CIR"):

	#compute mannings flow in full pipe
	A = xarea(shape, diameter, height, width)
	Rh = hydraulicRadius(shape, diameter, height, width)
	n = getMannings(shape, diameter)
	k = (1.49 / n) * math.pow(Rh, 0.667) * A

	Q = k * math.pow(slope/100.0, 0.5)

	return Q
