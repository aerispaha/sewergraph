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

    G1 = G.copy()

    for n in nx.topological_sort(G1):
        #loop through each and accumulate area with its immediate
        #upstream nodes. where there's a split, apply the split fraction
        area = sum(get_node_values(G1, [n], 'Shape_Area')) / 43560.0
        for p in G1.predecessors(n):
            area += G1.node[p]['total_area_ac']
            if 'flow_split_frac' in G1[p][n]:
                area *= G1[p][n]['flow_split_frac']

        G1.node[n]['total_area_ac'] = area

    for u,v,d in G1.edges_iter(data=True):

        #collect the upstream nodes, sum area
        up_nodes = nx.ancestors(G1, u) | set({u}) #upstream nodes
        acres = G1.node[v]['total_area_ac']

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
            path_len = sum([G2[u][v][parameter] for u,v in pairwise(path)])
            if path_len > longest_len:
                tc_path = path
                longest_len = path_len

    return tc_path

def analyze_downstream(G):
    """
    """
    G1 = G.copy()
    start_node = nx.topological_sort(G1, reverse=True)[0]
    for u,v,d in G1.edges_iter(data=True):
        descendants = []
        rates = []
        for path in nx.all_simple_paths(G1, source=v, target=start_node):

            #descendants = nx.descendants(G1, u)
            descendants += [G1[u][v] for u,v in pairwise(path)]
            rates += [(e['phs_rate'], e['FACILITYID']) for e in descendants]

        if descendants:
            sorted_rates = sorted(rates)
            d['descendants'] = [e['FACILITYID'] for e in descendants]
            d['limiting_rate'], d['limiting_sewer'] = sorted_rates[0]

    return G1

def analyze_flow_splits(G):

    """
    loop through nodes, find nodes with more than 1 outflow (flow split)
    tag the immediately downstream sewers as flow splitters
    find the resolving node by finding the closest downstream node with
    mulitple inflows and more than one path from the splitter to resolver
    """

    G1 = G.copy()

    #iterate through nodes having more than one out degree
    splitters = [(n, deg) for n, deg in G1.out_degree_iter() if deg > 1]
    for splitter, out_degree in splitters:

        #find the where the split is resolved
        candidates = [m for m in nx.descendants(G1, splitter)
                      if G1.in_degree(m) > 1]

        #find nodes that have mulitple paths connecting to split node
        resolver_paths = []
        for r in candidates:
            paths = list(nx.all_simple_paths(G1, source=splitter, target=r))
            if len(paths) > 1:
                resolver_paths += paths

        #find the shortest of all paths connecting the splitter & resolver
        #any longer paths are extensions along the same split.
        if resolver_paths:
            shortest = sorted(resolver_paths, key = len)[0]
            resolver = shortest[-1] #last node of the shortest path = resolver
            up_edges = [(up, resolver) for up in G1.predecessors_iter(resolver)]
            #tag the nodes and edges accordingly
            G1.node[resolver]['up_splitter'] = splitter
            G1.node[splitter]['dn_resolver'] = resolver
            for u,v in up_edges:
                G1[u][v]['flow_resolve'] = 'Y'
                G1[u][v]['split_node'] = splitter

        #record which segments are downstream of this node
        dwn_edges = [(splitter, dn) for dn in G1.successors_iter(splitter)]
        G1.node[splitter]['flow_split'] = splitter
        G1.node[splitter]['flow_split_edges'] = dwn_edges

        #tag the flow split sewers
        total_capacity = max(sum([G1[u][v]['capacity'] for u,v in dwn_edges]),1)
        for u,v in dwn_edges:
            G1[u][v]['flow_split'] = 'Y'
            if G1.in_degree(u)==0:
                G1[u][v]['flow_split'] = 'summet'
            G1[u][v]['flow_split_frac'] = G1[u][v]['capacity'] / total_capacity




    return G1
