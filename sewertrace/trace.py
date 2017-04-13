import networkx as nx
import pandas as pd
from helpers import (pairwise, visualize, open_file,
                     clean_network_data, get_node_values)
from hhcalculations import philly_storm_intensity, hhcalcs_on_network
from resolve_data import resolve_geom_slope_gaps
import os

class SewerNet(object):
    def __init__(self, shapefile, boundary_conditions=None, run=True):

        """
        Sewer network data wrapper that performs hydraulic and hydrologic (H&H)
        calculations on each sewer segment. H&H calculations use the Rational
        Method and the Manning equation to compute peak runoff and sewer
        capacity.

        Parameters
        ----------
        shapefile : string
            path to directory containing a shapefile(s) of spatial sewer
            data and manhole data (or other point spatial data) that connect
            sewers and contain a Shape_Area (square feet) field representing the drainage
            area attributed to that node. The Shape_Area field may be obtained
            by creating Theissen polygons around each point and joining the
            polygons' Shape_Area to the manhole attributes.

        boundary_conditions : dict of dicts
            additional data to join to point shapefiles based on a GUID
            (FACILITYID) key. The value of each GUID key is another dict
            containing the data that will be joined to the matching node.
            additional_area and travel_time are currently supported keys.

        run : bool, default True
            whether the 'hydrologic_calcs_on_sewers' and 'analyze_downstream'
            should be run upon instantiation.
        """

        self.shapefile_path = shapefile
        G = nx.read_shp(shapefile)

        #clean up the network (rm unecessary DataConv fields, isolated nodes)
        G = clean_network_data(G)
        G = nx.convert_node_labels_to_integers(G)
        G = resolve_geom_slope_gaps(G)

        #perform travel time and capacity calcs
        G = hhcalcs_on_network(G)

        #id flow split sewers and calculate split fractions
        G = analyze_flow_splits(G)

        if boundary_conditions is not None:
            add_boundary_conditions(G, boundary_conditions)
        self.boundary_conditions = boundary_conditions

        #accumulate drainage areas
        G = accumulate_area(G)

        #accumulating travel times
        G = accumulate_travel_time(G)
        self.G = G

        self.top_nodes = [n for n,d in G.in_degree_iter() if d == 0]
        self.terminal_nodes = [n for n,d in G.out_degree_iter() if d == 0]

        self.nbunch = None

        if run:
            self.G = hydrologic_calcs_on_sewers(self.G)
            self.G = analyze_downstream(self.G)

    def run_hydrology(self, nbunch=None, catchment_min=0.0, pipe_types=None):

        filtered_nodes = []
        #print 'running hydrologic calcs...'
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
        self.G = hydrologic_calcs_on_sewers(self.G, nbunch)

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
                'Shape_Leng', 'PIPE_TYPE']

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
                    #print 'adding data to {}'.format(fid)
                    d.update(data[fid])

def hydrologic_calcs_on_sewers(G, nbunch=None):
    G1 = G.copy()

    for u,v,d in G1.edges_iter(data=True, nbunch=nbunch):

        #grab the upstream node's total and direct area,
        #and apply any flow split fraction
        split_frac = d.get('flow_split_frac', 1)
        acres =     (G1.node[u]['total_area_ac'] * split_frac)
        direct_ac = (G1.node[u].get('Shape_Area',0) / 43560.0) * split_frac

        #grab the tc and path from the upstream node
        tc_path = G1.node[u]['tc_path']
        tc = G1.node[u]['tc']
        intensity = philly_storm_intensity(tc) #in/hr
        peakQ = 0.85 * intensity * acres # q = C*I*A, (cfs)

        #store values in the edge data (sewer reach)
        d['upstream_area_ac'] = acres
        d['direct_area_ac'] = direct_ac
        d['tc_path'] = tc_path
        d['tc'] = tc
        d['intensity'] = intensity
        d['peakQ'] = peakQ

        #compute the capacity fraction (hack prevent div/0)
        d['capacity_fraction'] = round(peakQ / max(d['capacity'], 1.0)*100)
        d['phs_rate'] = d['capacity'] / max(acres, 0.1) #prevent div/0 (FIX!!)

        #retain networkx up/down node information
        d['up_node'] = u
        d['dn_node'] = v

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

    while traversing the topologically sorted network, accumulate the list of
    upstream tc nodes for each subsequent node. This builds the tc_path param so
    we don't have to do any further tc computation.
    """

    G1 = G.copy()

    #assign inlet time of concentration
    for n, d in G1.nodes_iter(data=True):
        if G1.in_degree(n) == 0 and 'tc' not in d:
            #top of shed node, won't overwrite
            #boundary condition with tc param already set
            d['tc'] = 3 #minutes
            d['tc_path'] = n #to hold list of tc path nodes in descendants

    for n in nx.topological_sort(G1):
        #the current node tc
        tc = sum(get_node_values(G1, [n], ['tc']))
        path = get_node_values(G1, [n], ['tc_path']) #this is a copy, right?

        #create 2d array with the tc of any upstream edge + node pair, and the
        #precedessors' list of tc_path member nodes
        upstream_tc_options = [(G1[p][n]['travel_time'] +
                                G1.node[p]['tc'],
                                G1.node[p]['tc_path'])
                                for p in G1.predecessors(n)]

        if len(upstream_tc_options) > 0:
            #2d array gets sorted by tc, descending
            upstream_tc_options.sort(reverse=True)
            tc += upstream_tc_options[0][0]
            path += upstream_tc_options[0][1] + [n]
            # path.append(tc_nodes)

        G1.node[n]['tc'] = tc
        G1.node[n]['tc_path'] = path

    return G1


def analyze_downstream(G, nbunch=None, in_place=False, terminal_nodes=None):
    """
    Assign terminal nodes to each node in the network, then find the limiting
    sewer reach between each node and its terminal node.
    """
    if not in_place:
        G1 = G.copy()
    else:
        G1 = G
    if terminal_nodes is None:
        terminal_nodes = [n for n,d in G1.out_degree_iter() if d == 0]

    #assign terminal node(s) to each node
    #print 'finding terminal nodes...'
    for n in terminal_nodes:
        for a in nx.ancestors(G1, n) | set({n}):
            if 'terminal_nodes' in G1.node[a]:
                G.node[a]['terminal_nodes'] += [n]
            else:
                G.node[a]['terminal_nodes'] = [n]

    #print 'finding limiting sewers...'
    for tn in terminal_nodes:
        G1.node[tn]['limiting_rate'] = 9999
        G1.node[tn]['limiting_sewer'] = None

        for p in G1.predecessors(tn):
            G1[p][tn]['limiting_rate'] = G1[p][tn]['phs_rate']

    for n in nx.topological_sort(G1, reverse=True):
        dn_node_rates = [(G1.node[s]['limiting_rate'],
                          G1.node[s]['limiting_sewer']) for s in G1.successors(n)]
        dn_edge_rates = [(G1[n][s]['phs_rate'],
                          G1[n][s]['FACILITYID']) for s in G1.successors(n)]
        dn_rates = dn_node_rates + dn_edge_rates

        if len(dn_rates) > 0:
            sorted_rates = sorted(dn_rates)
            rate, fid = sorted_rates[0]
            G1.node[n]['limiting_rate'] = rate
            G1.node[n]['limiting_sewer'] = fid
            G1[n][s]['limiting_rate'] = rate
            G1[n][s]['limiting_sewer'] = fid

    return G1

def analyze_flow_splits(G):

    """
    loop through nodes, find nodes with more than 1 outflow (flow split)
    tag the immediately downstream sewers as flow splitters
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
