################################################################################
# Module: save_load.py
# Description: Save and load sewer networks to/from disk
# License: MIT, see full license in LICENSE.txt
# Web: https://github.com/aerispaha/sewergraph
################################################################################


import networkx as nx
import pandas as pd
from sewergraph import generate_facility_id


def to_swmm5_dataframes(G):
    """
    return an dict of dataframes for junctions, conduits, and coordinates
    elements in a SWMM5 inp
    """
    self.G = assign_inverts(self.G)

    #JUNCTIONS
    df = gdf_from_graph(G, return_type='nodes')
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
    conduits = gdf_from_graph(G, return_type='edges')

    #shorten conduit id
    conduits.index = [i[1:7] for i in conduits.index]

    #rename duplicate FACILITYIDs
    cols=pd.Series(conduits.index)
    for dup in conduits.index.get_duplicates():
        cols[conduits.index.get_loc(dup)]=[dup+'.'+str(d_idx)
                                           if d_idx!=0 else dup
                                           for d_idx in range(
                                               conduits.index.get_loc(dup).sum()
                                               )]
    conduits.index=cols
    conduits = conduits[['up_node', 'dn_node', 'length']]
    conduits['ManningN'] = 0.013
    conduits['InletOffset'] = 0
    conduits['OutletOffset'] = 0
    conduits['InitFlow'] = 0.0001
    conduits['MaxFlow'] = 0

    #XSECTIONS
    xsect = self.conduits()
    xsect.index = cols
    xsect = xsect[['pipeshape', 'diameter', 'height', 'width']]
    shape_map = {'CIR':'CIRCULAR'}
    xsect = xsect.replace({'pipeshape':shape_map})
    xsect = xsect.rename(columns={'diameter':'Geom1', 'height':'Geom2', 'width':'Geom3',  'pipeshape':'Shape'})

    #shift the geoms for EGG shaped
    xsect.loc[xsect.Shape=='EGG', 'Geom1'] = xsect.loc[xsect.Shape=='EGG', 'Geom2']
    xsect.loc[xsect.Shape=='EGG', 'Geom2'] = xsect.loc[xsect.Shape=='EGG', 'Geom3']
    xsect.loc[xsect.Shape=='EGG', 'Geom3'] = 0
    xsect['Geom4'] = 0

    #convert to inches
    geoms = ['Geom1', 'Geom2', 'Geom3', 'Geom4']
    xsect[geoms] = xsect[geoms] / 12

    xsect['Barrels'] = 1

    return dict(
        conduits = conduits,
        junctions=junctions,
        coordinates = coordinates,
        xsections = xsect
    )

def swmm5_polygons_from_sheds(shed_df):
    """
    with a GeoDataFrame of polygons, create a dataframe
    with unstacked coordinates in the format of the SWMM5 inp
    [Polygons] section
    """
    polys = shed_df[:]
    polys.index = polys.Name
    polys = polys[['geometry']]

    def unstack_poly_coords(row):
        xys = []
        try:
            xys = [(x,y) for x,y in row.geometry.boundary.coords]
        except:
            xys = [(x,y) for x,y in row.geometry.convex_hull.boundary.coords]

        return xys

    polys['coords'] = polys.apply(lambda row: unstack_poly_coords(row), axis=1)
    polys['Subcatchment'] = polys.index
    stacked_xys = polys.set_index('Subcatchment').coords.apply(pd.Series).stack().reset_index(level=-1, drop=True)
    xys_df = pd.DataFrame(data=stacked_xys.values.tolist(), columns=['X-Coord', 'Y-Coord'], index=stacked_xys.index)

    return xys_df


def graph_from_shp(pth=r'test_processed_01', idcol='facilityid', crs={'init': 'epsg:4326'}):
    """
    Load a shapefile from disk and convert the node/edge attributes to
    correct data types.
    Parameters
    ----------
    pth : string
    Returns
    -------
    networkx digraph
    """

    try:
        from shapely import wkt
    except ImportError:
        raise ImportError('shapely module needed. get it here: '
                          'https://pypi.org/project/Shapely/')

    G = nx.read_shp(pth, )
    G.graph['crs'] = crs
    G = nx.convert_node_labels_to_integers(G, label_attribute='coords')

    for u, v, d in G.edges(data=True):

        # create a shapely line geometry object
        d['geometry'] = wkt.loads(d['Wkt'])
        d['length'] = d['geometry'].length

        # get rid of other geom formats
        del d['Wkb'], d['Wkt'], d['Json']

        # generate a uniq id if necessary
        if idcol not in d:
            d[idcol] = generate_facility_id()

    return G


def gdf_from_graph(G, return_type='edges'):
    '''create a GeoDataFrame from a sewergraph, G.'''
    try:
        import geopandas as gp
    except ImportError:
        raise ImportError('GeoPandas module needed. Download with conda: '
                          'conda install geopandas')

    if return_type == 'edges':
        df = nx.to_pandas_edgelist(G)
        return gp.GeoDataFrame(df, crs=G.graph['crs'])

    elif return_type == 'nodes':
        node_dict = {n: d for n, d in G.nodes(data=True)}
        df = pd.DataFrame(node_dict).T
        return gp.GeoDataFrame(df, crs=G.graph['crs'])
    else:
        print('incorrect return type. should be "edges" or "nodes".')


def graph_from_gdf(gdf):
    '''create a sewergraph, G, from a GeoDataFrame'''
    G = nx.from_pandas_edgelist(gdf, create_using=nx.DiGraph(), edge_attr=True)
    G.graph['crs'] = gdf.crs
    return G
