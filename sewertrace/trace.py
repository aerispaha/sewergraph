from itertools import tee, izip
import networkx as nx
import random
import matplotlib.pyplot as plt
from .helpers import data_from_adjacent_node, pairwise
from hhcalculations import philly_storm_intensity

def upstream_accumulate_all(G, parameter='Shape_Area'):
    """
    compute the accumulation (sum) of the given parameter of all upstream nodes
    for each node in the network, G. (currently vars suggest upstream area)

    Note: This is essentially brute forcing the network, and could be more
    intelligently algorithmized e.g.: start at top or bottom and track which
    tributaries have/haven't been traversed.
    """

    #keep track of where flow splits are observed
    splitnodes = []
    G1 = G.copy()

    for u,v,d in G1.edges_iter(data=True):

        #collect the upstream nodes, sum area
        up_nodes = nx.ancestors(G1, u) | set({u}) #upstream nodes
        areas = get_node_values(G1, up_nodes, 'Shape_Area')
        acres = sum(areas) / 43560.0

        #find path and calculate tc (sum trave)
        tc_path = find_tc_path(G1, u, 'travel_time')
        tc = sum([G1[u][v]['travel_time'] for u,v in pairwise(tc_path)]) + 3.0
        intensity = philly_storm_intensity(tc) #in/hr
        peakQ = 0.85 * intensity * acres # q = C*I*A, (cfs)

        #store values in the edge data (sewer reach)
        d['upstream_area_ac'] = acres
        d['tc_path'] = tc_path
        d['tc'] = tc + d['travel_time']
        d['intensity'] = intensity
        d['peakQ'] = peakQ

        #compute the capacity fraction (hack prevent div/0)
        d['capacity_fraction'] = round(peakQ / max(d['capacity'], 1.0)*100)
        d['phs_rate'] = d['capacity'] / max(acres, 0.1) #prevent div/0 (FIX!!)

        #retain networkx up/down node information
        d['up_node'] = u
        d['dn_node'] = v
        d['up_fids'] = get_node_values(G1, up_nodes, 'FACILITYID')

        #remove extraneous data from sewer edges
        rmkeys = ['DownStream','LinerDate','LifecycleS','LinerType','Wkb','Wkt']
        [d.pop(k, None) for k in rmkeys]

    return G1
    
def upstream_accumulate(G, node, parameter=None):
    """
    sum of the values of each upstream manhole's parameter value,
    from the given node.
    """

    #return a set of the upstream nodes, including the current node
    upstream_nodes = nx.ancestors(G, node) | set({node})

    #find any flow splits
    splits = flow_splits(G)

    if parameter is None:
        upstream_count = [1 for n in upstream_nodes if 'Depth' in G.node[n]]
        return sum(upstream_count)

    upstream_vals = [G.node[n][parameter] for n in upstream_nodes
                     if parameter in G.node[n]]

    return sum(upstream_vals)

def get_node_values(G, nodes, parameter):
    """return a list of values in nodes having the parameter"""

    #if the parameter is in the node, return its value
    upstream_vals = [G.node[n][parameter] for n in nodes
                     if parameter in G.node[n]]
    return upstream_vals

def find_tc_path(G, start_node=None, parameter='length'):
    """
    find the path with the largest accumulation of the parameter (e.g. travel
    time). Return a list of nodes along this path
    """

    if start_node is None:
        start_node = nx.topological_sort(G, reverse=True)[0]

    #subset graph to all upstream
    up_nodes = nx.ancestors(G, start_node) | set({start_node})
    G2 = nx.subgraph(G, up_nodes)

    top_nodes = [n for n in G2.nodes_iter() if G2.in_degree(n) == 0]

    tc_path, longest_len = [], 0
    for n in top_nodes:
        for path in nx.all_simple_paths(G2, source=n, target=start_node):
            path_len = sum([G2[u][v][parameter] for u, v in pairwise(path)])
            if path_len > longest_len:
                tc_path = path
                longest_len = path_len

    return tc_path

def flow_splits(G):

    splitnodes = []
    for n, d in G.nodes_iter(data=True):
        if G.out_degree(n) > 1:
            #node has more than one successor, may be a flow split
            if 'Subtype' in G.node[n] and G.node[n]['Subtype'] != 6:

                G.node[n]['flow_split_keys']={}
                #subtype = 6 is a summit manhole
                splitnodes.append(n)


    return splitnodes

def find_resolved_splits(G):
    splitters = flow_splits(G)
    results = {}
    for s in splitters:

        #look for descendants with more than one path between the splitter
        s_degree =  G.out_degree(s) #out degree, i.e. how many paths its split into

        resolve_paths = []
        multi_in_ds = [d for d in nx.descendants(G, s) if G.in_degree(d) > 1]
        SUB = nx.subgraph(G, nbunch=multi_in_ds)
        for desc in nx.topological_sort(SUB, reverse=True):
            paths = [p for p in nx.all_simple_paths(G, source=s, target=desc)]

            if len(paths) > 1 and len(paths) == s_degree:
                resolve_paths += paths
                break

        results.update({s:resolve_paths})

    return results
