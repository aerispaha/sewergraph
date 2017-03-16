"""
HELPER FUNCTIONS FOR TRACING OPERATIONS
"""
from itertools import tee, izip
import networkx as nx
import json
from geojson import Feature, LineString, Point, FeatureCollection


def data_from_adjacent_node(G, n, key='FACILITYID'):
    """can be the current node if it has the data key"""

    nodes = G.nodes(data=True)
    m, d = nodes[n]
    if key in d:
        return d[key]

    for n in nx.ancestors(G, n):
        m, d = nodes[n]
        if key in d:
            return d[key]
        else:
            print '{} not found in {}'.format(key, n)

def pairwise(iterable):
    """
    take a list and pair up items sequentially. E.g. list of nodes to
    list of edges between the nodes.

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def random_alphanumeric(n=6):
	import random
	chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	return ''.join(random.choice(chars) for i in range(n))

def write_geojson(G, filename=None, geomtype='linestring', inproj='epsg:2272'):

    try: import pyproj
    except ImportError:
        raise ImportError('pyproj module needed. get this package here: ',
                        'https://pypi.python.org/pypi/pyproj')

    #SET UP THE TO AND FROM COORDINATE PROJECTION
    pa_plane = pyproj.Proj(init=inproj, preserve_units=True)
    wgs = pyproj.Proj(proj='longlat', datum='WGS84', ellps='WGS84') #google maps, etc

    #ITERATE THROUGH THE RECORDS AND CREATE GEOJSON OBJECTS
    G1 = G.copy()
    features = []
    if geomtype == 'linestring':
        for u,v,d in G1.edges_iter(data=True):
            coordinates = json.loads(d['Json'])['coordinates']
            latlngs = [pyproj.transform(pa_plane, wgs, *xy) for xy in coordinates]
            del d['Wkb'], d['Json']
            geometry = LineString(latlngs)

            feature = Feature(geometry=geometry, properties=d)
            features.append(feature)

    if filename is not None:
        with open(filename, 'wb') as f:
            f.write(json.dumps(FeatureCollection(features)))
        return filename

    else:
        return FeatureCollection(features)

def visualize(G, filename):
    import geojson

    basemap_path = r'P:\06_Tools\sewertrace\basemaps\mapbox_base.html'


    #create geojson, find bbox and center
    geo_conduits = write_geojson(G)

    #get center point
    xs = [d['X_Coord'] for n,d in G.nodes_iter(data=True) if 'X_Coord' in d]
    ys = [d['Y_Coord'] for n,d in G.nodes_iter(data=True) if 'Y_Coord' in d]
    c = ((max(xs) + min(xs))/2 , (max(ys) + min(ys))/2)
    bbox = [(min(xs), min(ys)), (max(xs), max(ys))]


    with open(basemap_path, 'r') as bm:
        # filename = os.path.join(os.path.dirname(geocondpath), self.alt_report.model.name + '.html')
        with open(filename, 'wb') as newmap:
            for line in bm:
                if '//INSERT GEOJSON HERE ~~~~~' in line:
                    newmap.write('conduits = {};\n'.format(geojson.dumps(geo_conduits)))
                    # newmap.write('nodes = {};\n'.format(0))
                    # newmap.write('parcels = {};\n'.format(geojson.dumps(geo_parcels)))
            	if 'center: [-75.148946, 39.921685],' in line:
					newmap.write('center:[{}, {}],\n'.format(c[0], c[1]))
                if '//INSERT BBOX HERE' in line:
                    newmap.write('map.fitBounds([[{}, {}], [{}, {}]]);\n'.format(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))

                else:
					newmap.write(line)