from itertools import tee, izip
import networkx as nx
import random
import matplotlib.pyplot as plt
from .helpers import data_from_adjacent_node, pairwise


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

def upstream_accumulate(G, node, parameter=None):
    """
    sum of the values of each upstream manhole's parameter value,
    from the given node.
    """

    #return a set of the upstream nodes, including the current node
    upstream_nodes = nx.ancestors(G, node) | set({node})

    #find any flow splits
    splits = flow_splits(G)

    #partition nodes
    # split_flow_nodes = [n for n in ]

    if parameter is None:
        upstream_count = [1 for n in upstream_nodes if 'Depth' in G.node[n]]
        return sum(upstream_count)

    upstream_vals = [G.node[n][parameter] for n in upstream_nodes
                     if parameter in G.node[n]]

    return sum(upstream_vals)


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

    for n, d in G1.nodes_iter(data=True):
        upstream_area = upstream_accumulate(G1, n, parameter)
        G1.node[n]['upstream_area_ac'] = upstream_area/43560.0


    for u,v,d in G1.edges_iter(data=True):

        #get the downstream node's accumulated area
        area = G1.node[u]['upstream_area_ac']

        #set this as the edge's upstream area
        G1.edge[u][v]['upstream_area_ac'] = area

        #upstream tc
        longest_path, longest_len = tc_path(G1, v)
        G1.edge[u][v]['tc_length'] = longest_len

    return G1

def tc_path(G, start_node):

    #subset graph to all upstream
    up_nodes = nx.ancestors(G, start_node) | set({start_node})
    G2 = nx.subgraph(G, up_nodes)

    top_nodes = [n for n in G2.nodes_iter() if G2.in_degree(n) == 0]

    longest_path, longest_len = [], 0
    for n in top_nodes:
        for path in nx.all_simple_paths(G2, source=n, target=start_node):
            path_len = sum([G2[u][v]['length'] for u, v in pairwise(path)])
            if path_len > longest_len:
                longest_path = path
                longest_len = path_len

    return longest_path, longest_len

def longest_path(shp=r'P:\06_Tools\sewertrace\data\oxford', firstn=None, draw=True):
    """
    find the longest upstream path in a given scope. optionally, pass in a
    first node (by default, this starts at the topologically last node).

    All paths are reviewed between the start node and each of the topmost nodes
    (topmost = nodes having zero inflow, in_degree==0)
    """
    #read shp file
    G = nx.read_shp(shp)

    #position the coordinates adjacent to node index
    #(for some reason read_shp() names the nodes by their coords)
    pos = {i:k for i,k in enumerate(G.nodes())}

    #rename the nodes to their index location (0, 1, 2...)
    r = range(0,len(G.nodes()))
    mapping=dict(zip(G.nodes(), r))
    G1=nx.relabel_nodes(G,mapping)

    #set edge weights based on length
    weights = []
    for u,v,d in G1.edges_iter(data=True):
        l = d['length']
        G1[u][v]['weight'] = l
        weights.append(l/100)

    #find all terminal nodes (nodes at top of watershed)
    topnodes = [x for x in G1.nodes_iter() if G1.out_degree(x)==1
                and G1.in_degree(x)==0]
    botnodes = [x for x in G1.nodes_iter() if G1.out_degree(x)==0
                and G1.in_degree(x)>0]

    #find the starting node if not passed in
    if firstn is None:
        firstn = nx.topological_sort(G1, reverse=True)[0]#[-1]

    #grab the FACILITYID of the start node (or a nearby node if first isn't a manhole with a FID)
    start_fid = data_from_adjacent_node(G1, firstn)

    #find the longest path
    longest_path, longest_len = [], 0
    split_paths = []
    for n in topnodes:
        #loop through each possible path between current top node and shed outlet
        #typically this should be only one path, unless a flow split exists
        npaths = len([p for p in nx.all_simple_paths(G1, source=n, target=firstn)]) #probably an unecessary performance drag
        for path in nx.all_simple_paths(G1, source=n, target=firstn):

            #calculate the length of this path
            path_len = sum([G1[u][v]['length'] for u, v in pairwise(path)])
            if path_len > longest_len:
                longest_path = path
                longest_len = path_len

            if npaths > 1:
                split_paths.append(path)


    longest_path_nodes = longest_path #nx.dag_longest_path(G1)
    longest_path_edges = [(u,v) for u,v in pairwise(longest_path_nodes)]

    if draw is False:
        return (G1, pos, longest_path_nodes)


    fig = plt.figure()
    fig.suptitle('Longest Upstream Path From Node {}\nFID = {}'.format(firstn,start_fid), fontsize=14, fontweight='bold')
    fig.text(0.95, 0.01, 'Longest path = {}ft'.format(round(longest_len,1)),
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=15)

    #draw the overall network
    nx.draw(G1, pos=pos, node_size=0, alpha=0.8, node_color='grey')

    # #draw the longest mh nodes that have attribute data
    mhs = [n for n in nx.ancestors(G1, u) if 'Depth' in G1.node[n]]
    nx.draw_networkx_nodes(G1,pos,
                       nodelist=mhs,
                       node_color='g',
                       node_size=30,
                   alpha=0.8)

    #highlight the start node
    nx.draw_networkx_nodes(G1,pos,
                       nodelist=[firstn],
                       node_color='g',
                       node_size=300,
                   alpha=0.8)

    nx.draw_networkx_edges(G1,pos, width=3, edgelist=longest_path_edges, edge_color='b')

    #draw possible flow splits
    for p in split_paths:
        edges = [(u,v) for u,v in pairwise(p)]
        nx.draw_networkx_edges(G1, pos, width=1, edgelist=edges, edge_color='r',
                               style='dotted')
#     nx.draw_networkx_edges(G1,pos, width=weights)
    # nx.draw_networkx_labels(G1,pos,font_size=14,font_family='sans-serif')#, labels=dict(zip(longest_path_nodes, longest_path_nodes)))

#     edge_labels=dict([((u,v,),round(d['weight'], 0)) for u,v,d in G1.edges_iter(data=True)])
#     nx.draw_networkx_edge_labels(G1,pos, edge_labels)
    return (G1, pos, longest_path_nodes, longest_path_edges)
