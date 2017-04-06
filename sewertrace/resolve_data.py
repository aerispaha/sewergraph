from itertools import chain
from sewertrace import helpers
import networkx as nx
def search(G, u, v, search_depth, length=0, data=[], el_var = 'ELEVATION_'):

    if search_depth == 0:
        return 'nothing found'

    similar_predecessors = [p for p in G.predecessors(u)
                            if G[p][u]['LABEL'] == G[u][v]['LABEL']]

    for i in similar_predecessors:
        #for i in G.predecessors_iter(u):
        # print "searching {}'s precedessor {}".format(u, i)

        pred_edge = G[i][u]
        length += pred_edge['Shape_Leng']


        pred_node = G.node[i]
        #if G.in_degree(i)<=1:
        if pred_node.get(el_var, 0) != 0:
            #el_var found in node
            # print pred_node[el_var]
            data += [(pred_node['FACILITYID'], i, pred_node[el_var], length)]

        search(G, i, u, search_depth-1)

    return data

def recommend_slope(G, u, v, el_var='ELEVATION_'):

    res = search(G, u, v, search_depth=10, el_var=el_var)
    rep_edges =  helpers.pairwise([r[1] for r in res]) #upstream repr edges
    slopes = []
    wt_slopes = []
    total_len = 0
    for t, s in rep_edges:
        #traverse the paths because there may be nodes in between nodes with el data
        for path in nx.all_simple_paths(G, source=s, target=t):
            path_len = sum([G[u][v]['Shape_Leng'] for u,v in helpers.pairwise(path)])
            up_el, dn_el = G.node[s][el_var], G.node[t][el_var]
            slope = (up_el-dn_el) / path_len
            total_len += path_len
            slopes.append(slope)
            wt_slopes.append((up_el-dn_el))

    if total_len != 0:
        return sum([s for s in wt_slopes if s > 0]) / total_len #length weighted average slope of upstream edges
    else:
        return None

def resolve_slope_gaps(G, nbunch=None):
    """
    find sewers with missing slope data and attempt to infer slope from adjacent
    sewers and manholes
    """

    G1 = G.copy()
    # zero_slope_sewers = list(set([(d['FACILITYID']) for  u,v,d, in
    #                               net.G.edges_iter(data=True)
    #                               if d['Slope'] == 0.0]))
    #
    # nbunch = list(set(list(chain.from_iterable(zero_slope_sewers))))

    for u,v,d in G1.edges_iter(data=True, nbunch=nbunch):

        #find bad slope edge
        if d['Slope'] == 0:
            # print 'attempting to resolve {}'.format((u,v))
            s = recommend_slope(G1, u, v, 'ELEVATION_')
            if s is not None:
                d['calculated_slope'] = s * 100.0
                # print d['FACILITYID'], 'calc slope', s * 100.0


    return G1
