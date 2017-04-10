import networkx as nx
import pandas as pd
from helpers import (pairwise, visualize, open_file,
                     clean_network_data, get_node_values)
from hhcalculations import philly_storm_intensity, hhcalcs_on_network
from resolve_data import resolve_slope_gaps,resolve_geom_slope_gaps
import os

class SewerNet(object):
    def __init__(self, shapefile, boundary_conditions=None):

        self.shapefile_path = shapefile
        print 'reading shapefile...'
        G = nx.read_shp(shapefile)

        #clean up the network (rm unecessary DataConv fields, isolated nodes)
        print 'resolving gaps...'
        G = clean_network_data(G)
        G = nx.convert_node_labels_to_integers(G)
        G = resolve_geom_slope_gaps(G)

        #perform travel time and capacity calcs
        print 'hhcacls...'
        G = hhcalcs_on_network(G)

        #id flow split sewers and calculate split fractions
        print 'analyzing flow splits...'
        G = analyze_flow_splits(G)

        if boundary_conditions is not None:
            print 'adding boundary conditions...'
            add_boundary_conditions(G, boundary_conditions)
        self.boundary_conditions = boundary_conditions

        print 'accumulating drainage areas...'
        G = accumulate_area(G)

        print 'accumulating travel times...'
        G = accumulate_travel_time(G)
        self.G = G

        self.top_nodes = [n for n in G.nodes_iter() if G.in_degree(n) == 0]
        self.terminal_nodes = [n for n in G.nodes_iter() if G.out_degree(n) == 0]

        self.nbunch = None

    def run_hydrology(self, nbunch=None, catchment_min=0.0, pipe_types=None):

        filtered_nodes = []
        print 'running hydrologic calcs...'
        #filter nodes given catchment_min and pipe_types
        for u,v,d, in self.G.edges_iter(data=True, nbunch=nbunch):
            if self.G.node[u]['total_area_ac'] >= catchment_min:
                if pipe_types is not None:
                    if str(d['PIPE_TYPE']) in pipe_types:
                        filtered_nodes += [u,v]
                else:
                    filtered_nodes += [u,v]

        #run the hydrologic calculations on sewers meeting filter criteria
        nbunch = set(filtered_nodes)
        self.nbunch = nbunch
        self.G = hydrologic_calcs_on_sewers(self.G, nbunch, self.top_nodes)

    def conduits(self):
        """
        return the networkx network edges as a Pandas Dataframe
        """
        fids = [d['FACILITYID'] for u,v,d in self.G.edges(data=True)]
        data = [d for u,v,d in self.G.edges(data=True)]
        df = pd.DataFrame(data=data, index=fids)
        cols = ['LABEL', 'Slope', 'capacity', 'peakQ','tc', 'intensity',
                'upstream_area_ac', 'phs_rate','limiting_rate','limiting_sewer',
                'Diameter','Height', 'Width', 'Year_Insta', 'FACILITYID',
                'Shape_Leng']

        return df[cols]

    def to_map(self, filename=None, startfile=True):

        if filename is None:
            filename = os.path.join(shapefile_path, 'map.html')

        visualize(self.G.subgraph(self.nbunch), filename, self.G)

        if startfile:
            open_file(filename)

def add_boundary_conditions(G, data):
    """
    add additional data to nodes in the sewer_net.G. Do this
    before running the accumulate_area
    """
    for n,d in G.nodes_iter(data=True):
        if 'FACILITYID' in d:
            for fid in data.keys():
                if fid in d['FACILITYID']:
                    print 'adding data to {}'.format(fid)
                    d.update(data[fid])

def hydrologic_calcs_on_sewers(G, nbunch=None, top_nodes=None):
    G1 = G.copy()

    if top_nodes is None:
        top_nodes = [n for n in G1.nodes_iter() if G1.in_degree(n) == 0]

    for u,v,d in G1.edges_iter(data=True, nbunch=nbunch):

        #collect the upstream nodes, sum area
        up_nodes = nx.ancestors(G1, u) | set({u}) #upstream nodes
        acres = G1.node[u]['total_area_ac']
        acres *= d.get('flow_split_frac', 1) #split the area if necessary

        #find path and calculate tc (sum trave)
        #tc_path = find_tc_path(G1, u, 'travel_time', top_nodes)
        tc_path = find_tc_path_by_walking(G1, u)
        # tc = sum([G1[u][v]['travel_time'] for u,v in pairwise(tc_path)]) + 3.0
        # tc += sum([G.node[n].get('travel_time', 0) for n in tc_path]) #boundary tc
        tc = G1.node[u]['tc']
        intensity = philly_storm_intensity(tc) #in/hr
        peakQ = 0.85 * intensity * acres # q = C*I*A, (cfs)

        #store values in the edge data (sewer reach)
        d['upstream_area_ac'] = acres
        d['tc_path'] = tc_path
        d['tc'] = tc #+ d['travel_time']
        d['intensity'] = intensity
        d['peakQ'] = peakQ

        #compute the capacity fraction (hack prevent div/0)
        d['capacity_fraction'] = round(peakQ / max(d['capacity'], 1.0)*100)
        d['phs_rate'] = d['capacity'] / max(acres, 0.1) #prevent div/0 (FIX!!)

        #retain networkx up/down node information
        d['up_node'] = u
        d['dn_node'] = v
        # d['up_fids'] = get_node_values(G1, up_nodes, 'FACILITYID')

    return G1

def accumulate_area(G):

    """
    loop through each node and accumulate area with its immediate
    upstream nodes. where there's a split, apply the split fraction to
    downstream edges.
    """
    G1 = G.copy()

    for n in nx.topological_sort(G1):
        area = sum(get_node_values(G1, [n], ['Shape_Area', 'additional_area']))
        area = area / 43560.0 #to acres
        for p in G1.predecessors(n):
            area += G1.node[p]['total_area_ac']
            if 'flow_split_frac' in G1[p][n]:
                area *= G1[p][n]['flow_split_frac']

        G1.node[n]['total_area_ac'] = area
    return G1

def accumulate_travel_time(G):
    """
    loop through each node and accumulate the travel time with its immediate
    upstream nodes and edges. where there are multiple precedessors, choose the
    upstream node + edge pair with the maximum travel time.
    """

    G1 = G.copy()

    #assign inlet time of concentration
    for n, d in G1.nodes_iter(data=True):
        if G1.in_degree(n) == 0 and 'tc' not in d:
            #top of shed node, won't overwrite
            #boundary condition with tc param already set
            d['tc'] = 3 #minutes

    for n in nx.topological_sort(G1):
        #the current node tc
        tc = sum(get_node_values(G1, [n], ['tc']))

        #add the max travel time of any upstream edge + node pair
        upstream_tc_options = [G1[p][n]['travel_time'] + G1.node[p]['tc']
                               for p in G1.predecessors(n)]
        if len(upstream_tc_options) > 0:
            tc += max(upstream_tc_options)

        G1.node[n]['tc'] = tc

    return G1

def find_tc_path_by_walking(G, start_node):
    """
    starting at a node, walk up the predecessors, always choosing the path
    with the maximum travel time at each junction of multiple sewers.

    This is slow for some reason. :(

    TRY LOOKING FOR all_simple_paths. FIRST CHECK CONNECTIVITY

    """

    G1 = G.copy()
    preds = G1.predecessors(start_node)
    path = []

    while len(preds) > 0:

        #list of tuples (tc, predecessor)
        pred_tcs = [(G1.node[p]['tc'], p) for p in  preds]
        pred_tcs.sort(reverse=True) #max tc at begining
        tc, selected_node = pred_tcs[0]
        path.append(selected_node)

        #restart search at the selected_node
        preds = G1.predecessors(selected_node)

    return path

def find_tc_path(G, start_node=None, parameter='length', top_nodes=None):
    """
    find the path with the largest accumulation of the parameter (e.g. travel
    time). Return a list of nodes along this path
    """

    if start_node is None:
        start_node = nx.topological_sort(G, reverse=True)[0]

    #subset graph to all upstream
    up_nodes = nx.ancestors(G, start_node) | set({start_node})
    G2 = nx.subgraph(G, up_nodes)

    #if top_nodes is None:
    top_nodes = [n for n in G2.nodes_iter() if G2.in_degree(n) == 0]


    tc_path, longest_len = [], 0
    for n in top_nodes:
        if n in G2:
            for path in nx.all_simple_paths(G2, source=n, target=start_node):
                path_len = sum([G2[u][v][parameter] for u,v in pairwise(path)])

                #add any added boundary conditions on nodes
                path_len += sum([G2.node[m].get(parameter, 0) for m in path])

                if path_len > longest_len:
                    tc_path = path
                    longest_len = path_len

    return tc_path

def analyze_downstream(G, nbunch=None, in_place=False):
    """
    """
    if not in_place:
        G1 = G.copy()
    else:
        G1 = G

    start_node = nx.topological_sort(G1, reverse=True, nbunch=nbunch)[0]
    for u,v,d in G1.edges_iter(data=True,nbunch=nbunch):
        descendants = []
        rates = []
        for path in nx.all_simple_paths(G1, source=v, target=start_node):

            #descendants = nx.descendants(G1, u)
            descendants += [G1[u][v] for u,v in pairwise(path)]
            rates += [(e['phs_rate'], e['FACILITYID']) for e in descendants
                      if 'phs_rate' in e]

        if descendants:
            sorted_rates = sorted(rates)
            # d['descendants'] = [e['FACILITYID'] for e in descendants]
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
