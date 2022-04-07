import networkx as nx
from .helpers import pairwise, get_node_values
from .hhcalculations import philly_storm_intensity


def hydrologic_calcs_on_sewers(G, nbunch=None, return_period=0):
    G1 = G.copy()

    for u, v, k, d in G1.edges(data=True, nbunch=nbunch, keys=True):
        # grab the upstream node's total and direct area,
        # and apply any flow split fraction
        split_frac = d.get('flow_split_frac', 1)
        direct_ac = (G1[u][v][k].get('local_area', 0) / 43560.0) * split_frac
        acres = (G1.nodes[u]['cumulative_area'] * split_frac / 43560.0) + direct_ac
        C = G1.nodes[u].get('runoff_coefficient', 0.85)  # direct area
        Cwt = G1.nodes[u].get('runoff_coefficient_weighted', 0.85)

        G1.nodes[u]['runoff_coefficient'] = C  # set it if its not there

        # grab the tc and path from the upstream node
        tc_path = G1.nodes[u]['tc_path']
        tc = G1.nodes[u]['tc']
        intensity = philly_storm_intensity(tc, return_period)  # in/hr
        peakQ = Cwt * intensity * acres  # q = C*I*A, (cfs)

        # store values in the edge data (sewer reach)
        d['upstream_area_ac'] = acres
        d['local_area_ac'] = direct_ac
        d['tc_path'] = tc_path
        d['tc'] = tc
        d['intensity'] = intensity
        d['peakQ'] = peakQ
        d['runoff_coefficient'] = C
        d['runoff_coefficient_weighted'] = Cwt
        d['CA'] = G1.nodes[u].get('CA', None)

        # compute the capacity fraction (hack prevent div/0)
        d['capacity_fraction'] = peakQ / max(d['capacity'], 1.0)
        d['capacity_per_ac'] = d['capacity'] / max(acres, 0.1)  # prevent div/0 (FIX!!)

        # retain networkx up/down node information
        d['up_node'] = u
        d['dn_node'] = v

    return G1


def accumulate_downstream(G, accum_attr='local_area', cumu_attr_name=None,
                          split_attr='flow_split_frac'):
    """
    pass through the graph from upstream to downstream and accumulate the value
    an attribute found in nodes and edges, and assign the accumulated value
    as a new attribute in each node and edge.

    Where there's a flow split, apply an optional split fraction to
    coded in the upstream edge.
    """
    G1 = G.copy()

    if cumu_attr_name is None:
        cumu_attr_name = 'cumulative_{}'.format(accum_attr)

    for n in nx.topological_sort(G1):

        # grab value in current node
        attrib_val = G1.nodes[n].get(accum_attr, 0)

        # sum with cumulative values in upstream nodes and edges
        for p in G1.predecessors(n):

            for k, v in G1[p][n].items():

                # add cumulative attribute val in upstream node, apply flow split fraction
                attrib_val += G1.nodes[p][cumu_attr_name] * G1[p][n][k].get(split_attr, 1)

                # add area routed directly to upstream edge/sewer
                attrib_val += G1[p][n][k].get(accum_attr, 0)

                # store cumulative value in upstream edge
                G1[p][n][k][cumu_attr_name] = attrib_val

        # store cumulative attribute value in current node
        G1.nodes[n][cumu_attr_name] = attrib_val

    return G1


def propogate_weighted_C(G, gsi_capture={}):
    """
    loop through each node and propogate the weighted C from the top to bottom
    of the shed. where there's a flow split, apply the split fraction to
    coded in the upstream edge (based on relative sewer capacity).
    """
    G1 = G.copy()

    for n in nx.topological_sort(G1):
        area = sum(get_node_values(G1, [n], ['local_area', 'additional_area']))
        C = G1.nodes[n].get('runoff_coefficient', 0.85)
        area = area / 43560.0  # to acres
        CA = C * area

        # set runoff_coefficient if not already set
        G1.nodes[n]['runoff_coefficient'] = C

        for p in G1.predecessors(n):
            pred = G1.nodes[p]  # upstream node
            # area += pred['cumulative_area'] * G1[p][n].get('flow_split_frac', 1)
            CA += pred['CA'] * G1[p][n].get('flow_split_frac', 1.0)

            # add area routed directly to sewer
            CA += G1[p][n].get('local_area', 0) * G1[p][n].get('runoff_coefficient', 0.85)

        # G1.nodes[n]['cumulative_area'] = area
        node = G1.nodes[n]
        node['CA'] = CA

        # apply GSI capture data at prescribed nodes
        if n in gsi_capture:
            frac = gsi_capture[n]['fraction']
            gsi_C = gsi_capture[n]['C']
            tot_area = node['cumulative_area']
            CA = ((1.0 - frac) * tot_area * C + frac * tot_area * gsi_C)
            node['CA'] = CA
            node['GSI Capture'] = gsi_capture[n]

        if node['cumulative_area'] > 0:
            node['runoff_coefficient_weighted'] = CA / node['cumulative_area']
        else:
            node['runoff_coefficient_weighted'] = C

    return G1


def accumulate_travel_time(G):
    """
    loop through each node and accumulate the travel time with its immediate
    upstream nodes and edges. where there are multiple precedessors, choose the
    upstream node + edge pair with the maximum travel time.

    while traversing the topologically sorted network, accumulate the list of
    upstream tc nodes for each subsequent node. This builds the tc_path param so
    we don't have to do any further tc computation.
    """

    G1 = G.copy()

    # assign inlet time of concentration
    for n, d in G1.nodes(data=True):
        if G1.in_degree(n) == 0 and 'tc' not in d:
            # top of shed node, won't overwrite
            # boundary condition with tc param already set
            d['tc'] = 3  # minutes
            d['tc_path'] = n  # to hold list of tc path nodes in descendants

    for n in nx.topological_sort(G1):
        # the current node tc
        tc = sum(get_node_values(G1, [n], ['tc']))
        path = get_node_values(G1, [n], ['tc_path'])  # this is a copy, right?

        # create 2d array with the tc of any upstream edge + node pair, and the
        # precedessors' list of tc_path member nodes
        upstream_tc_options = [(G1[p][n]['travel_time'] +
                                G1.nodes[p]['tc'],
                                G1.nodes[p]['tc_path'])
                               for p in G1.predecessors(n)]

        if len(upstream_tc_options) > 0:
            # 2d array gets sorted by tc, descending
            upstream_tc_options.sort(reverse=True)
            tc += upstream_tc_options[0][0]
            path += upstream_tc_options[0][1] + [n]
            # path.append(tc_nodes)

        G1.nodes[n]['tc'] = tc
        G1.nodes[n]['tc_path'] = path

    return G1


def analyze_downstream(G, nbunch=None, in_place=False, terminal_nodes=None,
                       parameter='capacity_per_ac'):
    """
    Assign terminal nodes to each node in the network, then find the limiting
    sewer reach between each node and its terminal node.
    """
    if not in_place:
        G1 = G.copy()
    else:
        G1 = G
    if terminal_nodes is None:
        terminal_nodes = [n for n, d in G1.out_degree() if d == 0]

    # find limiting sewers
    for tn in terminal_nodes:
        G1.nodes[tn]['limiting_rate'] = 9999
        G1.nodes[tn]['limiting_sewer'] = None

        for p in G1.predecessors(tn):
            edge = G1[p][tn]
            if isinstance(edge, nx.classes.coreviews.AtlasView):
                for fid, ed in edge.items():
                    ed['limiting_rate'] = ed[parameter]

            else:
                G1[p][tn]['limiting_rate'] = G1[p][tn][parameter]

    for n in list(reversed(list(nx.topological_sort(G1)))):
        dn_node_rates = [(G1.nodes[s]['limiting_rate'],
                          G1.nodes[s]['limiting_sewer']) for s in G1.successors(n)]
        dn_edge_rates = [(G1[n][s][parameter],
                          G1[n][s]['facilityid']) for s in G1.successors(n)]
        dn_rates = dn_node_rates + dn_edge_rates

        if len(dn_rates) > 0:
            sorted_rates = sorted(dn_rates)
            rate, fid = sorted_rates[0]
            G1.nodes[n]['limiting_rate'] = rate
            G1.nodes[n]['limiting_sewer'] = fid
            for s in G1.successors(n):
                G1[n][s]['limiting_rate'] = rate
                G1[n][s]['limiting_sewer'] = fid

                # BUG this isn't assigning the right limiting sewer to sewers
                # right at the split

    return G1


def assign_inflow_ratio(G, inflow_attr='TotalInflowV'):
    '''
    find junctions with multiple inflows and assign relative
    contribution ratios to each upstream edge, based on the
    ratio of the inflow_attr.

    This assumes the inflow_attr is a node attribute that needs
    to be assigned to each edge
    '''

    G2 = G.copy()

    # first, write the TotalInflowV to each downstream edge
    for n, inflow in G2.nodes(data=inflow_attr):
        for s in G2.successors(n):
            for k, v in G2[n][s].items():
                G2[n][s][k][inflow_attr] = inflow

    # iterate through nodes with multiple inflows and assign relative contribution
    # ratio to each upstream edge
    junction_nodes = [n for n, d in G2.in_degree() if d > 1]
    for j in junction_nodes:

        # calculate total inflow, filter out any Nones
        inflows = [inflow for _, _, inflow in G2.in_edges(j, data=inflow_attr)]
        total = sum([_f for _f in inflows if _f])

        # calculate relative contribution
        for u, v, k, inflow in G2.in_edges(j, data=inflow_attr, keys=True):
            G2[u][v][k]['relative_contribution'] = 1  # default
            if total != 0:
                G2[u][v][k]['relative_contribution'] = float(inflow) / float(total)

    return G2


def relative_outfall_contribution(G):
    '''
    calculate the relative contribution of node J to each
    downstream outfall. This function creates a dictionary of
    outfalls and relative contributions within each node of
    the graph, G.
    '''

    G1 = G.copy()
    # assign outfall contrib dicts to terminal nodes and edges
    for tn in [n for n, d in G1.out_degree() if d == 0]:
        G1.nodes[tn]['outfall_contrib'] = {tn: 1.0}
        for p in G1.predecessors(tn):
            for k, v in G1[p][tn].items():
                G1[p][tn][k]['outfall_contrib'] = {tn: 1.0}

    G1inv = G1.reverse()
    for j in nx.topological_sort(G1inv):

        # retrieve outfall contrib dict for j, or an empty dict
        of_contrib_j = G1inv.nodes[j].get('outfall_contrib', {})
        G1inv.nodes[j]['outfall_contrib'] = of_contrib_j
        for s in G1inv.predecessors(j):

            # retrieve outfall contrib dict for edge sj, or an empty dict
            of_contrib_sj = G1inv[s][j].get('outfall_contrib', {})
            for k, v in G1inv[s][j].items():
                G1inv[s][j][k]['outfall_contrib'] = of_contrib_j

            S = G1inv.nodes[s]
            # print (s, S)
            for OF, w_SOF in list(S['outfall_contrib'].items()):
                # get weight of node J w.r.t. OF by multiplying the
                # weight of S w.r.t OF by any inflow ratio to a junction
                # via edge JS. Store this in the outfall contrib dict in
                # node J and edge JS
                for k, v in G1inv[s][j].items():
                    w_JOF = w_SOF * G1inv[s][j][k].get('relative_contribution', 1)
                    of_contrib_j.update({OF: w_JOF})

            G1inv.nodes[j]['outfall_contrib'].update(of_contrib_j)
            for k, v in G1inv[s][j].items():
                G1inv[s][j][k]['outfall_contrib'].update(of_contrib_j)

    return G1inv.reverse()


def analyze_flow_splits(G, split_frac_attr='capacity'):
    """
    loop through nodes, find nodes with more than 1 outflow (flow split)
    tag the immediately downstream edges as flow splitters and calculate a
    flow split ratio to apply to each of the downstream edges.
    """

    G1 = G.copy()

    # iterate through nodes having more than one out degree
    splitters = [(n, deg) for n, deg in G1.out_degree() if deg > 1]
    for splitter, out_degree in splitters:

        # record which segments are downstream of this node
        dwn_edges = [(splitter, dn) for dn in G1.successors(splitter)]
        G1.nodes[splitter]['flow_split'] = splitter
        G1.nodes[splitter]['flow_split_edges'] = dwn_edges

        # tag the flow split sewers
        total_capacity = max(sum([G1[u][v][split_frac_attr] for u, v in dwn_edges]), 1)
        for u, v in dwn_edges:
            G1[u][v]['flow_split'] = 'Y'
            if G1.in_degree(u) == 0:
                G1[u][v]['flow_split'] = 'summet'
                G1.nodes[u]['flow_split'] = 'summet'
            G1[u][v]['flow_split_frac'] = G1[u][v][split_frac_attr] / total_capacity

    return G1


def map_to_lower_res_graph(G1, G2, rm_nodes=None, return_agg=False):
    '''
    given a skeletonized graph G2 benchmarked to a baseline
    graph G1, generate a map of nodes removed from G1 to their closest
    nodes in G2
    '''

    node_map = {}
    agg_map = {}
    if rm_nodes is None:
        rm_nodes = [n for n in G1 if n not in G2]
    elif any([n not in G1 for n in rm_nodes]):
        raise ('WHOA rm_node not found in G1')

    for n in rm_nodes:
        # node is removed in newer model. Find the closest
        # downstream node inflows can be redirected to (a node that
        # exists in model2)
        found_dn = 0
        searched_count = 0
        for dn in nx.dfs_preorder_nodes(G1, n):
            # print ('n:{} down: {}'.format(n, dn))
            # if downstream node the new model, save and break
            searched_count += 1
            if dn in G2:
                node_map.update({n: dn})
                agg_map.setdefault(dn, set()).add(n)
                # print ('mapped: {}'.format({n:dn}))
                found_dn += 1

                break
        if found_dn == 0:
            node_map.update({n: 'UNMATCHED'})
            agg_map.setdefault('UNMATCHED', set()).add(n)
            # print ('{} unmatched after searching {} (last searched: {})'.format(n,
            #                                                            searched_count,
            #                                                            dn))

    # update with nodes to keep
    # node_map.update({str(n):str(n) for n in G1 if n in G2})
    # agg_map.update({str(n):{str(n)} for n in G1 if n in G2})

    if return_agg:
        return agg_map
    else:
        return node_map


def find_edge(G, facilityid):
    '''find an edge given a facilityid'''
    for u, v, fid in G.edges(data='facilityid'):
        if fid == facilityid:
            return (u, v)


def set_flow_direction(G1, out):
    """
    Set the flow direction in a MultiDiGraph.

    Logic summary:
    For each node in an undirected copy of G,
    find edges in the simple short paths from n to all outs
    that don't exist in G, then reverse them
    """
    H1 = G1.to_undirected()

    rev_edges = []
    for n in H1.nodes():
        if nx.has_path(H1, n, out):
            for path in nx.shortest_simple_paths(H1, n, out):
                for u, v in list(pairwise(path)):
                    if G1.has_edge(u, v) is False:
                        rev_edges.append((v, u))

    G2 = G1.copy()
    for u, v in set(rev_edges):

        d = G2[u][v].copy()
        G2.remove_edge(u, v)
        # print (u,v, d)
        G2.add_edge(v, u)
        for k, val in list(d.items()):
            G2[v][u][k] = val

    return G2
