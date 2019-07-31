"""
HELPER FUNCTIONS FOR TRACING OPERATIONS
"""
from itertools import tee
import networkx as nx
import os, sys, subprocess
import pandas as pd
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
           sum_param='length'):

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


def transform_projection(G, to_crs='epsg:4326'):
    '''
    change the coordinate projection of Shapely "geometry" attributes in edges
    and nodes in the graph
    '''
    from functools import partial
    try:
        from shapely.ops import transform
        import pyproj
    except ImportError:
        raise ImportError('pyproj and shapely modules needed. get them here: ',
                          'https://pypi.org/project/pyproj/, '
                          'https://pypi.org/project/Shapely/')

    # set up the projection parameters
    st_plane = pyproj.Proj(G.graph['crs'], preserve_units=True)
    wgs = pyproj.Proj(init=to_crs)  # google maps, etc
    project = partial(pyproj.transform, st_plane, wgs)

    # apply transform to edge and node geometry attributes
    for u, v, geometry in G.edges(data='geometry'):
        if geometry:
            G[u][v]['geometry'] = transform(project, geometry)

    for n, geometry in G.nodes(data='geometry'):
        if geometry:
            G.node[n]['geometry'] = transform(project, geometry)

    G.graph['crs'] = to_crs
