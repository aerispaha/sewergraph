"""
HELPER FUNCTIONS FOR TRACING OPERATIONS
"""
from itertools import tee
import networkx as nx
from networkx.readwrite import json_graph
import json
from geojson import Feature, LineString, Point, FeatureCollection
import os, sys, subprocess
import pandas as pd
import geopandas as gp
import uuid

def generate_facility_id(length = 8):
    """
    Generate a facility id (random alpha numeric)
    """
    return str(uuid.uuid4()).upper()[:8]

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

def data_from_adjacent_node(G, n, key='facilityid'):
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
            print(('{} not found in {}'.format(key, n)))

def get_node_values(G, nodes, parameters):
    """return a list of values in nodes having the parameter"""

    #if the parameter is in the node, return its value
    upstream_vals = [G.node[n][p] for n in nodes
                      for p in parameters if p in G.node[n]]

    return upstream_vals

def subset(df, lb=None, ub=None, param='capacity_fraction',
           sum_param='Shape_Leng'):

    """
    subset a Dataframe between the provided lowerbound and upperbound (lb, ub)
    along the given parameter. optionally return a scalar sum of data within the
    sum_param
    """
    if lb is not None:
        df = df.loc[df[param] >= lb]
    if ub is not None:
        df = df.loc[df[param] < ub]

    if sum_param is not None:
        return df[sum_param].sum()
    else:
        return df

def clean_dict(mydict, keep_keys=None, rm_keys=None):
    """
    remove unwanted items in dicts
    """
    if keep_keys is not None:
        for k in list(mydict.keys()):
            if k not in keep_keys:
                del mydict[k]

    if rm_keys is not None:
        for k in list(mydict.keys()):
            if k in rm_keys:
                del mydict[k]

def rename_duplicates(series):
    #rename duplicate FACILITYIDs
    cols=pd.Series(series)
    for dup in series.get_duplicates():
        cols[series.get_loc(dup)]=[dup+'.'+str(d_idx)
                                           if d_idx!=0 else dup
                                           for d_idx in range(
                                               series.get_loc(dup).sum()
                                               )]
    return cols

def round_shapefile_node_keys(G):
    """
    nodes read in via nx.read_shp() are labeled as tuples representing the
    coordinates of each node in the shapefile. In order to join nodes who are
    very close (but not exactly the same), round the labels to the nearest
    integer. This works well for state plane coordinate systems.
    """
    mapping = {n:tuple([round(i, 2) for i in n]) for n in G.nodes()}
    return nx.relabel_nodes(G, mapping)

def clean_network_data(G):
    """
    remove unecessary fields from DataConv from the network, remove isolated
    nodes
    """
    G1 = G.copy()

    #remove isolated nodes
    G1.remove_nodes_from(nx.isolates(G1))

    for u,v,d in G1.edges(data=True):
        node_keeper_keys = ['X_Coord', 'Y_Coord','cumulative_area',
                            'local_area', 'facilityid', 'ELEVATION_', 'ELEVATIONI',
                            'FacilityNa', 'RASTERVALU']
        edge_keeper_keys = ['diameter', 'height','width', 'facilityid','Json',
                            'slope', 'Shape_Leng', 'Year_Insta', 'pipeshape',
                            'PIPE_TYPE', 'STICKERLIN', 'LABEL','ELEVATION_',
                            'ELEVATIONI','slope_calculated',
                            'slope_calculated_fids', 'local_area']
        clean_dict(G1.node[u], node_keeper_keys)
        clean_dict(G1.node[v], node_keeper_keys)
        clean_dict(d, edge_keeper_keys)

    return G1

def pairwise(iterable):
    """
    take a list and pair up items sequentially. E.g. list of nodes to
    list of edges between the nodes.

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return list(zip(a, b))

def random_alphanumeric(n=6):
	import random
	chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	return ''.join(random.choice(chars) for i in range(n))


def create_html_map(geo_layers, filename, G, basemap='mapbox_base.html'):
    """
    geo_layers: dict of dicts
        dictionary of layers and their geojson data
    """
    import geojson

    # This is the project root #HACK
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASEMAP_PATH = os.path.join(ROOT_DIR,'basemaps',basemap)
    # basemap_path = r'P:\06_Tools\sewertrace\basemaps\mapbox_base.html'

    #get center point
    gdf = gp.GeoDataFrame(nx.to_pandas_edgelist(G)[['geometry']])
    hull = gdf.unary_union.convex_hull
    c = (hull.centroid.x, hull.centroid.y)
    bbox = hull.bounds

    with open(BASEMAP_PATH, 'r') as bm:
        # filename = os.path.join(os.path.dirname(geocondpath), self.alt_report.model.name + '.html')
        with open(filename, 'wb') as newmap:
            for line in bm:
                if '//INSERT GEOJSON HERE ~~~~~' in line:
                    for lyr, geodata in list(geo_layers.items()):
                        jsondata = geojson.dumps(geodata)
                        newmap.write('{} = {};\n'.format(lyr, jsondata))

                    #write the network as a json object
                    # net_dict = json_graph.node_link_data(G)
                    edges = list([(u,v, {'facilityid':d}) for u,v,d in G.edges.data('facilityid')])
                    nodes = list(G.nodes())

                    # newmap.write('net_json = {};\n'.format(json.dumps(net_dict)))
                    newmap.write('edges = {};\n'.format(json.dumps(edges)))
                    newmap.write('nodes = {};\n'.format(json.dumps(nodes)))

                if 'center: [-75.148946, 39.921685],' in line:
                    newmap.write('center:[{}, {}],\n'.format(c[0], c[1]))
                if '//INSERT BBOX HERE' in line:
                    newmap.write('map.fitBounds([[{}, {}], [{}, {}]]);\n'
                                 .format(bbox[0], bbox[1], bbox[2],
                                         bbox[3]))

                else:
                    newmap.write(line)
