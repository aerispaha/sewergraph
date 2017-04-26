import networkx as nx
import pandas as pd
from helpers import (pairwise, visualize, open_file,
                     clean_network_data, get_node_values, round_shapefile_node_keys)
import helpers
from hhcalculations import philly_storm_intensity, hhcalcs_on_network
from resolve_data import resolve_geom_slope_gaps
from kpi import SewerShedKPI
import cost_estimates
import os

class SewerNet(object):
    def __init__(self, shapefile=None, G=None, boundary_conditions=None, run=True,
                 return_period=0, name=None):

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

        boundary_conditions : dict of dicts, default None
            additional data to join to point shapefiles based on a GUID
            (FACILITYID) key. The value of each GUID key is another dict
            containing the data that will be joined to the matching node.
            additional_area and travel_time are currently supported keys.

        run : bool, default True
            whether the 'hydrologic_calcs_on_sewers' and 'analyze_downstream'
            should be run upon instantiation.
        """
        if shapefile is not None:
            self.shapefile_path = shapefile
            G = nx.read_shp(shapefile)

            #clean up the network (rm unecessary DataConv fields, isolated nodes)
            G = clean_network_data(G)
            G = round_shapefile_node_keys(G)
            G = nx.convert_node_labels_to_integers(G, label_attribute='coords')
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
        self.name = name

        #summary calculations
        self.top_nodes = [n for n,d in G.in_degree_iter() if d == 0]
        self.terminal_nodes = [n for n,d in G.out_degree_iter() if d == 0]
        self.nbunch = None

        if run:
            self.G = hydrologic_calcs_on_sewers(self.G, return_period=return_period)
            self.G = analyze_downstream(self.G)

        self.kpi = SewerShedKPI(self)

    def subshed(self, outfall_node=None, outfall_fid=None, name=None):
        """
        return a SewerNet object with everything upstream of the outfall node
        given by node or FACILITYID
        """
        if outfall_node is not None:
            tn = outfall_node
        if outfall_fid is not None:
            tn = [n for n,d in self.G.nodes_iter(data=True) if 'FACILITYID' in d
                  and d['FACILITYID']==outfall_fid][0]

        #collect the nodes upstream of the terminal node, tn
        nbunch = nx.ancestors(self.G, tn) | set({tn})
        G = self.G.subgraph(nbunch).copy()

        return SewerNet(G=G, name=name)

    def estimate_sewer_replacement_costs(self, target_cap_frac=1.0):
        """
        calculate the required replacement size of all sewers to meet the
        target_cap_frac
        """
        cost_estimates.replacements_for_capacity(self.G, target_cap_frac)
        df = self.conduits()
        millions = df[df.replacement_cost > 0].replacement_cost.sum() / 10**6
        return millions

    def update_hydraulics(self, return_period=0):
        """
        re run the hydraulic calculations on the network
        """
        self.G = hhcalcs_on_network(self.G)
        self.G = analyze_flow_splits(self.G)
        # self.G = accumulate_travel_time(self.G)
        self.G = hydrologic_calcs_on_sewers(self.G, return_period=return_period)
        self.G = analyze_downstream(self.G)
        self.kpi = SewerShedKPI(self)

    def assign_runoff_coefficient(self, C, manhole_fids=None):
        """
        set the runoff coefficient of each node to the given C. optionaly
        isolate manholes by FACILITYID
        """
        for n,d in self.G.nodes_iter(data=True):
            d['runoff_coefficient'] = C

    def conduits(self):
        """
        return the networkx network edges as a Pandas Dataframe
        """
        fids = [d['FACILITYID'] for u,v,d in self.G.edges(data=True)]
        data = [d for u,v,d in self.G.edges(data=True)]
        df = pd.DataFrame(data=data, index=fids)

        return df

    def nodes(self):
        """
        return the networkx network nodes as a Pandas Dataframe
        """
        # fids = [d['FACILITYID'] for u,v,d in self.G.edges(data=True)]
        data = [d for n,d in self.G.nodes(data=True)]
        df = pd.DataFrame(data=data, index=self.G.nodes())
        return df

    def to_map(self, filename=None, startfile=True, phs_area=False):

        if filename is None:
            filename = os.path.join(shapefile_path, 'map.html')

        if phs_area:
            phs_rates = [{'FACILITYID':d['FACILITYID'],
                          'limiting_rate':d['limiting_rate']}
                         for n, d in self.G.nodes_iter(data=True)
                         if 'FACILITYID' in d]
            lyrs = dict(
                conduits = helpers.write_geojson(self.G),
                phs_sheds = phs_rates
            )
            helpers.create_html_map(lyrs, filename, self.G,'phs_sheds.html')
            print 'phs yea'
        else:
            lyrs = dict(conduits = helpers.write_geojson(self.G))
            helpers.create_html_map(lyrs, filename, self.G)


        # helpers.create_html_map(lyrs, filename, self.G)
        # visualize(self.G.subgraph(self.nbunch), filename, self.G)

        if startfile:
            open_file(filename)

    def to_swmm5_dataframes(self):
        """
        return an dict of dataframes for junctions, conduits, and coordinates
        elements in a SWMM5 inp
        """

        #JUNCTIONS
        df = self.nodes()
        df['MaxDepth'] = 0
        df['InitDepth'] = 0
        df['SurchargeDepth'] = 0
        df['PondedArea'] = 0
        node_cols = ['invert', 'MaxDepth', 'InitDepth', 'SurchargeDepth', 'PondedArea']
        junctions = df[node_cols]

        #COORDINATES
        df['x_coord'] = df.apply(lambda row: row.coords[0], axis=1)
        df['y_coord'] = df.apply(lambda row: row.coords[1], axis=1)
        coordinates = df[['x_coord', 'y_coord']]

        #CONDUITS
        conduits = self.conduits()

        #rename duplicate FACILITYIDs
        cols=pd.Series(conduits.index)
        for dup in conduits.index.get_duplicates():
            cols[conduits.index.get_loc(dup)]=[dup+'.'+str(d_idx)
                                               if d_idx!=0 else dup
                                               for d_idx in range(
                                                   conduits.index.get_loc(dup).sum()
                                                   )]
        conduits.index=cols
        conduits = conduits[['up_node', 'dn_node', 'Shape_Leng']]
        conduits['ManningN'] = 0.013
        conduits['InletOffset'] = 0
        conduits['OutletOffset'] = 0
        conduits['InitFlow'] = 0.0001
        conduits['MaxFlow'] = 0

        return dict(
            conduits = conduits,
            junctions=junctions,
            coordinates = coordinates,
        )

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

def hydrologic_calcs_on_sewers(G, nbunch=None, return_period=0):
    G1 = G.copy()

    for u,v,d in G1.edges_iter(data=True, nbunch=nbunch):

        #grab the upstream node's total and direct area,
        #and apply any flow split fraction
        split_frac = d.get('flow_split_frac', 1)
        acres =     (G1.node[u]['total_area_ac'] * split_frac)
        direct_ac = (G1.node[u].get('Shape_Area',0) / 43560.0) * split_frac
        C =  G1.node[u].get('runoff_coefficient', 0.85)
        G1.node[u]['runoff_coefficient'] = C #set it if its not there

        #grab the tc and path from the upstream node
        tc_path = G1.node[u]['tc_path']
        tc = G1.node[u]['tc']
        intensity = philly_storm_intensity(tc, return_period) #in/hr
        peakQ = C * intensity * acres # q = C*I*A, (cfs)

        #store values in the edge data (sewer reach)
        d['upstream_area_ac'] = acres
        d['direct_area_ac'] = direct_ac
        d['tc_path'] = tc_path
        d['tc'] = tc
        d['intensity'] = intensity
        d['peakQ'] = peakQ

        #compute the capacity fraction (hack prevent div/0)
        d['capacity_fraction'] = peakQ / max(d['capacity'], 1.0)
        d['capacity_per_ac'] = d['capacity'] / max(acres, 0.1) #prevent div/0 (FIX!!)

        #retain networkx up/down node information
        d['up_node'] = u
        d['dn_node'] = v

    return G1

def accumulate_area(G):

    """
    loop through each node and accumulate area with its immediate
    upstream nodes. where there's a flow split, apply the split fraction to
    coded in the upstream edge (based on relative sewer capacity).
    """
    G1 = G.copy()

    for n in nx.topological_sort(G1):
        area = sum(get_node_values(G1, [n], ['Shape_Area', 'additional_area']))
        area = area / 43560.0 #to acres

        for p in G1.predecessors(n):
            pred = G1.node[p] #upstream node
            area += pred['total_area_ac'] * G1[p][n].get('flow_split_frac', 1)

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

    #find limiting sewers
    for tn in terminal_nodes:
        G1.node[tn]['limiting_rate'] = 9999
        G1.node[tn]['limiting_sewer'] = None

        for p in G1.predecessors(tn):
            G1[p][tn]['limiting_rate'] = G1[p][tn]['capacity_per_ac']

    for n in nx.topological_sort(G1, reverse=True):
        dn_node_rates = [(G1.node[s]['limiting_rate'],
                          G1.node[s]['limiting_sewer']) for s in G1.successors(n)]
        dn_edge_rates = [(G1[n][s]['capacity_per_ac'],
                          G1[n][s]['FACILITYID']) for s in G1.successors(n)]
        dn_rates = dn_node_rates + dn_edge_rates

        if len(dn_rates) > 0:
            sorted_rates = sorted(dn_rates)
            rate, fid = sorted_rates[0]
            G1.node[n]['limiting_rate'] = rate
            G1.node[n]['limiting_sewer'] = fid
            for s in G1.successors(n):
                G1[n][s]['limiting_rate'] = rate
                G1[n][s]['limiting_sewer'] = fid

                #BUG this isn't assigning the right limiting sewer to sewers
                #right at the split



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
                G1.node[u]['flow_split'] = 'summet'
            G1[u][v]['flow_split_frac'] = G1[u][v]['capacity'] / total_capacity

    return G1
