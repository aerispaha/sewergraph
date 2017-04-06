"""
HELPER FUNCTIONS FOR TRACING OPERATIONS
"""
from itertools import tee, izip
import networkx as nx
from networkx.readwrite import json_graph
import json
from geojson import Feature, LineString, Point, FeatureCollection
import os, sys, subprocess

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener ="open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

def sum_params_in_nodes(G, nodes, parameter):
    """Sum the returned value of each parameter for all nodes
    having the parameter, in network G"""
    vals = [G.node[n][parameter] for n in nodes if parameter in G.node[n]]
    return sum(vals)

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
            geometry = LineString(latlngs)

            feature = Feature(geometry=geometry, properties=d)
            features.append(feature)

    if geomtype == 'point':
        for u, d in G1.nodes_iter(data=True):
            try:
                adjacent_n = G1[u].keys()[0]
                adjacent_edge =  G1[u][adjacent_n]
                coordinates = json.loads(adjacent_edge['Json'])['coordinates'][0]
                latlngs = [pyproj.transform(pa_plane, wgs, *xy) for xy in [coordinates]]

                geometry = Point(latlngs[0])

                feature = Feature(geometry=geometry, properties=d)
                features.append(feature)
            except:
                pass
    if filename is not None:
        with open(filename, 'wb') as f:
            f.write(json.dumps(FeatureCollection(features)))
        return filename

    else:
        return FeatureCollection(features)

def visualize(G, filename, full_G=None):
    import geojson

    # This is the project root #HACK
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASEMAP_PATH = os.path.join(ROOT_DIR,'basemaps','mapbox_base.html')
    # basemap_path = r'P:\06_Tools\sewertrace\basemaps\mapbox_base.html'


    #create geojson, find bbox and center
    geo_conduits = write_geojson(G)
    # geo_nodes = write_geojson(G, geomtype='point')

    #get center point
    xs = [d['X_Coord'] for n,d in G.nodes_iter(data=True) if 'X_Coord' in d]
    ys = [d['Y_Coord'] for n,d in G.nodes_iter(data=True) if 'Y_Coord' in d]
    c = ((max(xs) + min(xs))/2 , (max(ys) + min(ys))/2)
    bbox = [(min(xs), min(ys)), (max(xs), max(ys))]


    with open(BASEMAP_PATH, 'r') as bm:
        # filename = os.path.join(os.path.dirname(geocondpath), self.alt_report.model.name + '.html')
        with open(filename, 'wb') as newmap:
            for line in bm:
                if '//INSERT GEOJSON HERE ~~~~~' in line:
                    newmap.write('conduits = {};\n'.format(geojson.dumps(geo_conduits)))
                    # newmap.write('nodes = {};\n'.format(geojson.dumps(geo_nodes)))

                    #write the network as a json object
                    # net_dict = json_graph.node_link_data(G)
                    edges = G.edges()
                    nodes = G.nodes(data=True)
                    if full_G is not None:
                        #pass the entire network to jsnetworkx
                        edges = full_G.edges()
                        nodes = full_G.nodes(data=True)

                    # newmap.write('net_json = {};\n'.format(json.dumps(net_dict)))
                    newmap.write('edges = {};\n'.format(json.dumps(edges)))
                    newmap.write('nodes = {};\n'.format(json.dumps(nodes)))

            	if 'center: [-75.148946, 39.921685],' in line:
					newmap.write('center:[{}, {}],\n'.format(c[0], c[1]))
                if '//INSERT BBOX HERE' in line:
                    newmap.write('map.fitBounds([[{}, {}], [{}, {}]]);\n'.format(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))

                else:
					newmap.write(line)
