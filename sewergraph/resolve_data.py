import logging
import math

import networkx as nx
from .hhcalculations import slope_at_velocity


def preprocess_data(G):
    for u, v, d in G.edges(data=True):
        # normalize geometry coding
        geom = G[u][v]['pipeshape']
        diam = G[u][v]['diameter']
        h = G[u][v]['height']
        w = G[u][v]['width']

        # standardize unknowns
        if geom not in ['BOX', 'CIR', 'EGG'] or sum([_f for _f in [diam, h, w] if _f]
                                                    ) == 0:
            d['pipeshape'] = None
        elif (geom == 'CIR' and diam == 0 and h > 0 and w > 0):
            # if the geoms don't resemble a circle
            d['pipeshape'] = infer_from_dimensions(diam, h, w)
            d['inferred_geom'] = 'Y'
        elif (h == 0 and w == 0 and geom in ['EGG', 'BOX']):
            # if the geoms don't resemble a box/egg
            d['pipeshape'] = infer_from_dimensions(diam, h, w)
            d['inferred_geom'] = 'Y'

    for n, d in G.nodes(data=True):
        # normalize elevation data
        if 'ELEVATIONI' in d and d['ELEVATIONI'] == 0:
            del d['ELEVATIONI']  # = None


def infer_from_dimensions(diam, h, w):
    # see if we can infer from the attributes
    if diam > 0:
        return 'CIR'
    if h >= 60 and w > 0:
        return 'BOX'
    if h > 0 and w > 0 and h < 60:
        return 'EGG'


def resolve_geometry(G, u, v, search_depth=5):
    label = fid = None
    diam, h, w = G[u][v]['diameter'], G[u][v]['height'], G[u][v]['width']
    i = 0
    up, dn = u, v
    geom = infer_from_dimensions(diam, h, w)

    # find upstream edges that have the attribute, select the largest as the
    # representative sewer (most sewers will increase in size traversing downstrm)
    up_edges_data = dfs_edges_upstream_attributes(G, u, 'pipeshape')
    if len(up_edges_data) > 0:
        geoms = [(d['diameter'] + d['height'] + d['width'], i, j) for i, j, d in up_edges_data]
        geoms.sort(reverse=True)
        _, i, j = geoms[0]

        geom = G[i][j]['pipeshape']
        diam = G[i][j]['diameter']
        h = G[i][j]['height']
        w = G[i][j]['width']
        label = G[i][j]['LABEL']
        fid = G[i][j]['facilityid']

        if geom not in ['BOX', 'CIR', 'EGG']:
            geom = infer_from_dimensions(diam, h, w)

    # search downsteam for geom info
    i = 0
    while geom == None and search_depth > i:

        for dn in G.successors(v):
            # avoid modeling after a downstream collector
            geom = G[v][dn]['pipeshape']
            diam = G[v][dn]['diameter']
            h = G[v][dn]['height']
            w = G[v][dn]['width']
            label = G[v][dn]['LABEL']
            fid = G[v][dn]['facilityid']

            if geom not in ['BOX', 'CIR', 'EGG']:
                geom = infer_from_dimensions(diam, h, w)

        v = dn
        i += 1

    return geom, diam, h, w, label, fid


def dfs_nodes_attributes_iter(G, source=None, attribute=None, upstream=True,
                              null_val=None):
    """
    Produce node attributes in a depth-first-search (DFS) traversing
    upstream and terminating each search leg when finding a node having a
    non-null attribute.

    Based on Networkx dfs source

    NOTE: resolve slopes by dfs upstream and downstream finding the closest
    nodes with trusted elevation data having the largest cumulative drainage
    area.

    """
    if source is None:
        # produce edges for all components
        nodes = G
    else:
        # produce edges for components with source
        nodes = [source]
    visited = set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        if upstream:
            stack = [(iter(G.pred[start]), start)]
        else:
            stack = [(iter(G.succ[start]), start)]
        while stack:
            parents, child = stack[-1]
            try:
                parent = next(parents)
                node = G.nodes[parent]
                if attribute in node and node[attribute] is not null_val:
                    # by not appending to the search stack, this reach is no
                    # longer traversed
                    yield parent, node
                else:
                    # keep searching
                    if upstream:
                        stack.append((iter(G.pred[parent]), parent))
                    else:
                        stack.append((iter(G.succ[parent]), parent))

                if parent not in visited:
                    visited.add(parent)

            except StopIteration:
                # print edge['facilityid'], 'terminal', [i[1] for i in stack]
                stack.pop()


def dfs_nodes_attributes(G, source=None, attribute=None, upstream=True, null_val=None):
    res = dfs_nodes_attributes_iter(G, source, attribute, upstream, null_val=null_val)
    return list(res)


def dfs_edges_upstream_attributes_iter(G, source=None, attribute=None):
    """
    Produce edge attributes in a depth-first-search (DFS) traversing
    upstream and terminating each search leg when finding an edge having a
    non-null attribute.

    Based on Networkx dfs source
    """
    if source is None:
        # produce edges for all components
        nodes = G
    else:
        # produce edges for components with source
        nodes = [source]
    visited = set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(iter(G.pred[start]), start)]
        while stack:
            parents, child = stack[-1]
            try:
                parent = next(parents)
                edge = G[parent][child]
                if edge[attribute] is not None:
                    # by not appending to the search stack, this reach is no
                    # longer traversed
                    yield (parent, child, edge)
                else:
                    # keep searching
                    stack.append((iter(G.pred[parent]), parent))
                if parent not in visited:
                    visited.add(parent)

            except StopIteration:
                # print edge['facilityid'], 'terminal', [i[1] for i in stack]
                stack.pop()


def dfs_edges_upstream_attributes(G, source=None, attribute=None):
    res = dfs_edges_upstream_attributes_iter(G, source, attribute)
    return list(res)


def determine_slope_from_adjacent_inverts(G, u, v, data_key='ELEVATIONI'):
    # defaults
    i, j = u, v
    up_inv = dn_inv = 0
    length = 1
    slope = 0

    def adjacent_inv(G, n, data_key, upstream=True):

        # use the up/dn invs from the node having the highest accumulated area
        data = dfs_nodes_attributes(G, n, data_key, upstream)
        invs = [(d['cumulative_area'], d[data_key], n) for n, d in data]
        invs.sort(reverse=True)
        if len(invs) > 0:
            _, up_inv, ajd_n = invs[0]
            return up_inv, ajd_n
        else:
            return None, None

    # first, check if trusted invs exist in u or v, else find trusted inverts up/dwn
    if data_key in G.nodes[u]:
        up_inv, i = G.nodes[u][data_key], u
    else:
        up_inv, i = adjacent_inv(G, u, data_key, upstream=True)

    if data_key in G.nodes[v]:
        dn_inv, j = G.nodes[v][data_key], v
    else:
        dn_inv, j = adjacent_inv(G, v, data_key, upstream=False)

    # we we can't find an elevation upstream, search downstream
    if up_inv is None and dn_inv is not None:
        up_inv, i = dn_inv, j
        dn_inv, j = adjacent_inv(G, i, data_key, upstream=False)
    # we we can't find an elevation downstream, search upstream
    if dn_inv is None and up_inv is not None:
        dn_inv, j = up_inv, i
        up_inv, i = adjacent_inv(G, j, data_key, upstream=True)

    # calculate the slope
    if (up_inv and dn_inv) is not None:
        # get the length between the trusted inverts
        length = nx.shortest_path_length(G, source=i, target=j, weight='length')
        slope = (up_inv - dn_inv) / length
        return slope, i, j, length

    else:
        # print 'up_inv:{}, dn_inv:{}'.format(up_inv, dn_inv)
        return 0, u, v, 0


def resolve_slope_gaps(G):
    G1 = G.copy()
    slope_resolved_count = 0
    for u, v, d in G1.edges(data=True):
        if d['slope'] == 0 or math.isnan(d['slope']):
            slope, i, j, l = determine_slope_from_adjacent_inverts(G1, u, v)

            if slope < 0:
                # if negative slope assume slope for min design velocity
                height, width = d['height'], d['width']
                shape, diameter = d['pipeshape'], d['diameter']
                slope = slope_at_velocity(2.5, diameter, height, width, shape)

            d['slope'] = slope
            d['slope_source'] = [i, j]

            slope_resolved_count += 1
    logging.info(f'Resolved {slope_resolved_count} slope values.')
    return G1


def resolve_geom_gaps(G, nbunch=None):
    """
    find sewers with missing geom data and attempt to infer from adjacent
    sewers
    """

    G1 = G.copy()
    preprocess_data(G1)
    G1 = extend_elevation_data(G1)
    for u, v, d in G1.edges(data=True, nbunch=nbunch):

        if d['pipeshape'] not in ['BOX', 'CIR', 'EGG']:
            d['pipeshape'] = None  # overwrite, rid of 'UNK' issues

            # resolve geometry based on adjacent upstream sewer
            shape, diam, h, w, label, fid = resolve_geometry(G1, u, v)

            # overwrite attributes
            d['pipeshape'], d['diameter'] = shape, diam
            d['height'], d['width'], d['LABEL'] = h, w, label
            d['geometry_source'] = fid

    return G1


def extend_elevation_data(G, data_key='ELEVATIONI', null_val=0):
    """
    Using what trusted slope and elevation data exisits in the network,
    extend the"trusted" elevations where possible. This is accomplished by
    calculating new inverts upstream of nodes with trusted inverts connected
    by edges (sewers) with trusted slope values.
    """
    G1 = G.copy()

    topo_sorted_nodes = nx.topological_sort(G1)

    for n in list(reversed(list(topo_sorted_nodes))):
        # TRAVERSE THE TREE FROM BOTTOM TO TOP
        invert = None

        # assign the trusted invert
        if data_key in G1.node[n] and G1.node[n][data_key] != null_val:
            invert = G1.node[n][data_key]
            G1.node[n]['invert_trusted'] = invert

        if 'invert_trusted' in G1.node[n]:
            invert = G1.node[n]['invert_trusted']

        if invert is not None:
            for p in G1.predecessors(n):
                if G1[p][n]['slope'] != 0 and ('invert_trusted' and data_key) not in G1.node[p]:
                    # trusted slope, can calculate trusted invert
                    slope = G1[p][n]['slope'] / 100.0
                    length = G1[p][n]['length']
                    G1.node[p]['invert_trusted'] = invert + (slope * length)

    for n in topo_sorted_nodes:

        # TRAVERSE THE TREE FROM TOP TO BOTTOM
        invert = None

        # assign the trusted invert
        if data_key in G1.node[n] and G1.node[n][data_key] != null_val:
            invert = G1.node[n][data_key]

        if 'invert_trusted' in G1.node[n]:
            invert = G1.node[n]['invert_trusted']

        if invert is not None:
            for s in G1.successors(n):
                if G1[n][s]['slope'] != 0 and ('invert_trusted' and data_key) not in G1.node[s]:
                    # trusted slope, can calculate trusted invert
                    slope = G1[n][s]['slope'] / 100.0
                    length = G1[n][s]['length']
                    G1.node[s]['invert_trusted'] = invert - (slope * length)

    return G1


def elevation_change(G, s, t):
    """elevation difference between two nodes in graph, G"""
    length = G[s][t]['length']
    slope = G[s][t]['slope_used_in_calcs']
    delta = (slope / 100.0) * length
    return delta


def assign_inverts(G, data_key='ELEVATIONI'):
    """
    Assign invert values for each node in the graph, G by
    walking up the network and accumulating elevation based on
    sewer slope and length
    """

    G1 = G.copy()
    topo_sorted_nodes = nx.topological_sort(G1, reverse=True)
    terminal_nodes = [n for n, d in G1.out_degree_iter() if d == 0]

    # Assign starting El
    for tn in terminal_nodes:
        # set the terminal node invert by measuring the delta between
        # it and then nearest upstream node having elevation data
        G1.node[tn]['invert'] = 0

        # find tn position in the topologically sorted nodes
        i = topo_sorted_nodes.index(tn)

        while G1.node[tn]['invert'] == 0:
            delta = 0
            # loop through sorted nodes, starting at the tn position
            for n in topo_sorted_nodes[i:]:
                # depth first search to nearest node with trusted elevation data
                for u, v, direction in nx.edge_dfs(G1, n, 'reverse'):
                    delta += elevation_change(G1, u, v)
                    if 'invert_trusted' in G1.node[u]:
                        G1.node[tn]['invert'] = G1.node[u]['invert_trusted'] - delta
                        break

                if G1.node[tn]['invert'] != 0:
                    break

    for n in topo_sorted_nodes:
        for p in G1.predecessors(n):

            el_0 = G1.node[n]['invert']
            el_2 = el_0 + elevation_change(G1, p, n)

            # fill invert values where nodes already have trusted invert vals
            if G1.node[n].get('invert_trusted', 0) != 0:
                el_0 = G1.node[n]['invert_trusted']
                G.nodes[n]['invert'] = el_0
            if G1.node[p].get('invert_trusted', 0) != 0:
                el_2 = G1.node[p]['invert_trusted']

            G1.node[p]['invert'] = el_2

    return G1
