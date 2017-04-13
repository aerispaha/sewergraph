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
