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

    res = search(G, u, v, search_depth=3, el_var=el_var)
    print res
    rep_edges =  list(helpers.pairwise([r[1] for r in res]))#upstream repr edges
    slopes = []
    wt_slopes = []
    total_len = 0
    edges = []
    for t, s in rep_edges:
        #traverse the paths because there may be nodes in between nodes with el data
        for path in nx.all_simple_paths(G, source=s, target=t):
            path_len = sum([G[u][v]['Shape_Leng'] for u,v in helpers.pairwise(path)])
            edges += [G[u][v]['FACILITYID'] for u,v in helpers.pairwise(path)]
            up_el, dn_el = G.node[s][el_var], G.node[t][el_var]
            slope = (up_el - dn_el) / path_len
            total_len += path_len
            slopes.append(slope)
            wt_slopes.append((up_el-dn_el))

    if total_len != 0:
        #length weighted average slope of upstream edges
        weighed_avg = sum([s for s in wt_slopes if s > 0]) / total_len
        # fids = [G[v][u]['FACILITYID'] for u,v in rep_edges]
        # print 'hi ', edges
        return  (weighed_avg, edges)
    else:
        return (None, None) #this is dumb

def resolve_geom_slope_gaps(G, nbunch=None):
    """
    find sewers with missing geom data and attempt to infer from adjacent
    sewers
    """

    G1 = G.copy()


    for u,v,d in G1.edges_iter(data=True, nbunch=nbunch):

        if d['PIPESHAPE'] not in ['BOX', 'CIR', 'EGG']:
            d['PIPESHAPE'] = None #overwrite,  rid of 'UNK' issues

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

def resolve_slope_gaps(G, nbunch=None):
    """
    find sewers with missing slope data and attempt to infer slope from adjacent
    sewers and manholes
    """

    G1 = G.copy()


    for u,v,d in G1.edges_iter(data=True, nbunch=nbunch):

        #find bad slope edge
        if d['Slope'] == 0:
            # print 'attempting to resolve {}'.format((u,v))
            s, fids = recommend_slope(G1, u, v, 'ELEVATION_')
            if s is not None:
                d['calculated_slope'] = s * 100.0
                d['calculated_slope_fids'] = fids
                # print d['FACILITYID'], 'calc slope', s * 100.0, fids


    return G1
