from itertools import chain
from sewertrace import helpers
import networkx as nx


def resolve_geometry(G, u, v, search_depth=5):

    geom = label = fid = None
    diam, h, w = G[u][v]['Diameter'], G[u][v]['Height'], G[u][v]['Width']
    i = 0
    up, dn = u, v

    def infer_from_dimensions(diam, h, w):
        #see if we can infer from the attributes
        if diam > 0:
            return 'CIR'
        if h >= 60 and w > 0:
            return 'BOX'
        if h > 0 and w > 0 and h < 60:
            return 'EGG'


    geom = infer_from_dimensions(diam, h, w)

    while geom == None and search_depth > i:

        for up in G.predecessors(u):
            #avoid modeling after a downstream collector
            geom = G[up][u]['PIPESHAPE']
            diam = G[up][u]['Diameter']
            h = G[up][u]['Height']
            w = G[up][u]['Width']
            label = G[up][u]['LABEL']
            fid = G[up][u]['FACILITYID']

            if geom not in ['BOX', 'CIR', 'EGG']:
                geom = infer_from_dimensions(diam, h, w)

            # print '{}. up sew ({},{}), geom={}, fid = {}'.format(i, up, u, geom, fid)

        u = up
        i +=1

    #search downsteam for geom info
    i = 0
    while geom == None and search_depth > i:

        for dn in G.successors(v):
            #avoid modeling after a downstream collector
            geom = G[v][dn]['PIPESHAPE']
            diam = G[v][dn]['Diameter']
            h = G[v][dn]['Height']
            w = G[v][dn]['Width']
            label = G[v][dn]['LABEL']
            fid = G[v][dn]['FACILITYID']

            if geom not in ['BOX', 'CIR', 'EGG']:
                geom = infer_from_dimensions(diam, h, w)

        v = dn
        i +=1

    return geom, diam, h, w, label, fid

def resolve_slope(G, u, v, search_depth=10):

    i = up_slope = dn_slope = 0
    up, dn = u, v
    fids = []

    #search upstream for a slope
    while up_slope == 0 and search_depth > i:
        for up in G.predecessors(u):
            up_slope = G[up][u]['Slope']
            fids.append(G[up][u]['FACILITYID'])
            # print '{}  up_slope {}'.format(u, up_slope)

        u = up
        i +=1

    #search downsteam for a slope
    while dn_slope == 0 and search_depth > i:
        for dn in G.successors(v):
            dn_slope = G[v][dn]['Slope']
            fids.append(G[v][dn]['FACILITYID'])
            # print '{} dn slope {}'.format(v, dn_slope)

        v = dn
        i +=1

    return (up_slope, dn_slope, fids)

def resolve_geom_slope_gaps(G, nbunch=None):
    """
    find sewers with missing geom data and attempt to infer from adjacent
    sewers
    """

    G1 = G.copy()


    for u,v,d in G1.edges_iter(data=True, nbunch=nbunch):

        if d['PIPESHAPE'] not in ['BOX', 'CIR', 'EGG']:
            d['PIPESHAPE'] = None #overwrite, rid of 'UNK' issues

            #resolve geometry based on adjacent upstream sewer
            shape, diam, h, w, label, fid = resolve_geometry(G1, u, v)

            #overwrite attributes
            d['PIPESHAPE'], d['Diameter'] = shape, diam
            d['Height'], d['Width'], d['LABEL'] = h, w, label
            d['geometry_source'] = fid

        if d['Slope'] == 0:

            up_slope, dn_slope, fids = resolve_slope(G1, u, v)
            d['Slope'] = min(up_slope, dn_slope)
            d['slope_source'] = fids


    return G1

def assign_paths_to_data(G, data_key='ELEVATIONI', null_val=0):
    """
    for each node, determine which closest nodes upstream and downstream
    have an attribute equal to the data_key with a value not equal to the
    null_val
    """
    G1 = G.copy()
    topo_sorted_nodes = nx.topological_sort(G1, reverse=True)

    terminal_nodes = [n for n,d in G1.out_degree_iter() if d == 0]
    top_nodes = [n for n,d in G1.in_degree_iter() if d == 0]

    #Assign starting El
    for tn in terminal_nodes+top_nodes:
        #set the terminal nodes data
        G1.node[tn]['dn_nodes_with_data'] = []
        G1.node[tn]['up_nodes_with_data'] = []

    for n in topo_sorted_nodes:
        if data_key in G1.node[n] and G1.node[n][data_key] != null_val:
            G1.node[n]['dn_nodes_with_data'] = [n]

        else:
            G1.node[n]['dn_nodes_with_data'] = []
            for s in G1.successors(n):
                G1.node[n]['dn_nodes_with_data'] += G1.node[s]['dn_nodes_with_data']

    for n in list(reversed(topo_sorted_nodes)):
        if data_key in G1.node[n] and G1.node[n][data_key] != null_val:
            G1.node[n]['up_nodes_with_data'] = [n]
        else:
            G1.node[n]['up_nodes_with_data'] = []
            for p in G1.predecessors(n):
                G1.node[n]['up_nodes_with_data'] += G1.node[p]['up_nodes_with_data']

    return G1

def extend_elevation_data(G, data_key='ELEVATIONI', null_val=0):
    """
    calculate elevations of nodes where slopes exist along edges adjacent
    to nodes with elevation data
    """
    G1 = G.copy()
    topo_sorted_nodes = nx.topological_sort(G1, reverse=True)

    for n in topo_sorted_nodes:

        invert = None

        #assign the trusted invert
        if data_key in G1.node[n] and G1.node[n][data_key] != null_val:
            invert = G1.node[n][data_key]
            G1.node[n]['invert_trusted'] = invert

        if 'invert_trusted' in G1.node[n]:
            invert = G1.node[n]['invert_trusted']

        if invert is not None:
            for p in G1.predecessors(n):
                if G1[p][n]['Slope'] !=0 and ('invert_trusted' and data_key) not in G1.node[p]:
                    #trusted slope, can calculate trusted invert
                    slope = G1[p][n]['Slope'] / 100.0
                    length = G1[p][n]['Shape_Leng']
                    G1.node[p]['invert_trusted'] = invert + (slope * length)

    for n in list(reversed(topo_sorted_nodes)):
        invert = None

        #assign the trusted invert
        if data_key in G1.node[n] and G1.node[n][data_key] != null_val:
            invert = G1.node[n][data_key]

        if 'invert_trusted' in G1.node[n]:
            invert = G1.node[n]['invert_trusted']

        if invert is not None:
            for s in G1.successors(n):
                if G1[n][s]['Slope'] !=0 and ('invert_trusted' and data_key) not in G1.node[s]:
                    #trusted slope, can calculate trusted invert
                    slope = G1[n][s]['Slope'] / 100.0
                    lenght = G1[n][s]['Shape_Leng']
                    G1.node[n]['invert_trusted'] = invert - (slope * length)

    return G1

def elevation_change(G, s, t):
        """elevation difference between two nodes in graph, G"""
        l = G[s][t]['Shape_Leng']
        s = G[s][t]['slope_used_in_calcs']
        delta = (s/100.0) * l
        return delta

def assign_inverts(G):

    """
    Assign a invert values for each node in the graph, G by
    walking up the network and accumulating elevation based on
    sewer slope and length
    """

    G1 = G.copy()
    topo_sorted_nodes = nx.topological_sort(G1, reverse=True)
    terminal_nodes = [n for n,d in G1.out_degree_iter() if d == 0]

    #Assign starting El
    for tn in terminal_nodes:
        #set the terminal node invert by measuring the delta between
        #it and then nearest upstream node having elevation data
        G1.node[tn]['invert'] = 0

        #find tn position in the topologically sorted nodes
        i = topo_sorted_nodes.index(tn)

        while G1.node[tn]['invert'] == 0:
            delta = 0
            #loop through sorted nodes, starting at the tn position
            for n in topo_sorted_nodes[i:]:
                for p in G1.predecessors(n):
                    delta += elevation_change(G1, p, n)
                    if 'ELEVATIONI' in G1.node[p] and G1.node[p]['ELEVATIONI'] > 0:

                        G1.node[tn]['invert'] = G1.node[p]['ELEVATIONI'] - delta
                        print G1.node[p]['ELEVATIONI'], delta
                        break
                if G1.node[tn]['invert'] != 0:
                    break


    for n in topo_sorted_nodes:
        for p in G1.predecessors(n):

            el_0 = G1.node[n]['invert']
            el_2 = el_0 + elevation_change(G1,p,n)

            if G1.node[n].get('ELEVATIONI', 0) != 0:
                el_0 = G1.node[n]['ELEVATIONI']
                G.node[n]['invert'] = el_0
            if G1.node[p].get('ELEVATIONI', 0) != 0:
                el_2 = G1.node[p]['ELEVATIONI']

            G1.node[p]['invert'] = el_2

    return G1
