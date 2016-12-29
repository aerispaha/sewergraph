

from itertools import tee, izip
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def data_from_adjacent_node(G, n, key='FACILITYID'):
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
            print '{} not found in {}'.format(key, n)

def show_longest_path(shp=r'P:\06_Tools\trace\data\oxford', firstn=None):

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
    topnodes = [x for x in G1.nodes_iter() if G1.out_degree(x)==1 and G1.in_degree(x)==0]

    #find the starting node if not passed in
    if firstn is None:
        firstn = nx.topological_sort(G1)[-1]

    #grab the FACILITYID of the start node (or a nearby node if firstn isn't a manhole with a FID)
    start_fid = data_from_adjacent_node(G1, firstn)

    #find the longest path
    longest_path, longest_len = [], 0
    for n in topnodes:
        #loop through each possible path between current top node and shed outlet
        #typically this should be only one path, unless a flow split exists
        for path in nx.all_simple_paths(G1, source=n, target=firstn):

            #calculate the length of this path
            path_len = sum([G1[u][v]['length'] for u, v in pairwise(path)])
            if path_len > longest_len:
                longest_path = path
                longest_len = path_len

            print '{}-{} L={}'.format(path[0], path[-1], round(path_len, 1))

    longest_path_nodes = longest_path #nx.dag_longest_path(G1)
    longest_path_edges = [(u,v) for u,v in pairwise(longest_path_nodes)]

    fig = plt.figure()
    fig.suptitle('Longest Upstream Path From Node {}\nFID = {}'.format(firstn,start_fid), fontsize=14, fontweight='bold')
    fig.text(0.95, 0.01, 'Longest path = {}ft'.format(round(longest_len,1)),
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=15)

    #draw the overall network
    nx.draw(G1, pos=pos, node_size=100, alpha=0.8, node_color='grey')

    #draw the longest path
    nx.draw_networkx_nodes(G1,pos,
                       nodelist=longest_path_nodes,
                       node_color='b',
                       node_size=100,
                   alpha=0.8)

    #highlight the start node
    nx.draw_networkx_nodes(G1,pos,
                       nodelist=[firstn],
                       node_color='g',
                       node_size=300,
                   alpha=0.8)

    nx.draw_networkx_edges(G1,pos, width=3, edgelist=longest_path_edges, edge_color='b')
#     nx.draw_networkx_edges(G1,pos, width=weights)
#     nx.draw_networkx_labels(G1,pos,font_size=14,font_family='sans-serif')#, labels=dict(zip(longest_path_nodes, longest_path_nodes)))

#     edge_labels=dict([((u,v,),round(d['weight'], 0)) for u,v,d in G1.edges_iter(data=True)])
#     nx.draw_networkx_edge_labels(G1,pos, edge_labels)
    return G1
