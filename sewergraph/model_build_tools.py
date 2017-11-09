from shapely.geometry import MultiLineString, LineString
import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from rasterstats import zonal_stats, point_query
import ogr, osr, gdal
import os


def drainage_areas_from_sewers(sewersdf, SEWER_ID_COL):

    """
    create a GeoDataFrame of polygons representing sewersheds. Shed boundaries are
    created based on Voronoi polygons about each sewer segment.
    """
    #import into a GeoDataFrame
    #pth = r'trace_shapefiles'
    #sewersdf = gpd.read_file(pth, layer='sw_gravmains_2')

    #create a Shapely object
    sewer_shapes = MultiLineString([g for g in sewersdf.geometry])

    #array of points to be used in Voronoi generation
    pts = [
        (sewer.interpolate(pt, normalized=True).x, sewer.interpolate(pt, normalized=True).y)
        for pt in np.linspace(0.1, 1, 16)
        for sewer in sewer_shapes if sewer.length > 35
    ]

    #create a study area boundary to clip to Voronoi polygons to
    study_area_df = gpd.read_file(r'shapefiles', layer='study_area_bounds')
    study_area = study_area_df.loc[0, 'geometry']
    study_area_buff = study_area_df.convex_hull.buffer(distance=5000)
    border_pts = [xy for xy in study_area_buff.loc[0].boundary.coords]

    #create Voronoi object
    vor = Voronoi(pts+border_pts)

    #create shapely LineStrings of drainage areas boundaries
    drainage_bounds = [
        LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]

    #create a list of drainage area polygons and clip them (via intersection) to the study_area
    da_list = [da.intersection(study_area) for da in shapely.ops.polygonize(drainage_bounds)]

    #creat a MultiPloygon shapely object
    shed_pieces = shapely.geometry.MultiPolygon(da_list)

    #create GeoDataFrame of shed pieces
    shed_geoms = [g for g in shed_pieces.geoms]
    shed_areas_sf = [shed.area for shed in shed_geoms]
    sheds = gpd.GeoDataFrame(geometry=shed_geoms, data={'local_area':shed_areas_sf})

    #set crs and create a subshed id column
    sheds.crs = {'init':'epsg:2272'}
    sheds['SUBSHED_ID'] = sheds.index

    #spatially join the subsheds to the sewers, drop duplicates (sheds touching multiple sewers)
    # sewersdf = sewersdf[[SEWER_ID_COL, 'geometry']]
    sewer_sheds = gpd.sjoin(sheds, sewersdf, how='inner')
    sewer_sheds = sewer_sheds.drop_duplicates(subset='SUBSHED_ID')

    #dissolve by sewer FACILITYID, add area column
    sewer_sheds = sewer_sheds.dissolve(by=SEWER_ID_COL, aggfunc='sum', as_index=False)
    sewer_sheds = sewer_sheds[[SEWER_ID_COL, 'geometry']] #drop unnecessary cols

    return sewer_sheds


def overlay_proportion(row, df, overlay_field, overlay_category):
        """
        calculate the fraction of area covered by an overlay. input df is a
        GeoDataFrame resulting from an intersection of two mutlipolygon data sets.
        """

        ov_frac = 0 #default

        #isolate the study area based on input_field and input_id.
        #study_area_df = df.loc[df[input_field]==row[input_field]]
        study_area_df = df.loc[df['idx1']==row.name]
        study_area = study_area_df.geometry.area.sum()

        if study_area > 0:
            #isolate the overlay area based on the overlay_category.
            overlay_df = study_area_df.loc[study_area_df[overlay_field]==overlay_category]
            overlay_area = overlay_df.geometry.area.sum()

            #calculate the fraction of overlay area in the study area
            ov_frac = overlay_area / study_area

        return ov_frac

def calculate_overlay_fraction(zones, overlay, overlay_field='FCODE', overlay_category='Impervious'):
    """
    calculate the fraction of each overlay category present in each polyon in the zone
    GeoDataFrame. E.g. calculate imperviousness of sewersheds by passing in drainage areas
    and an impervious coverage layer.
    """

    #intersect the sheds and impervious cover spatial data
    overlay = overlay[[overlay_field, 'geometry']]
    ovdf = spatial_overlays(zones, overlay, how='intersection')


    ov_frac = zones.apply(lambda row:
                        overlay_proportion(row, ovdf, overlay_field, overlay_category),
                        axis=1)

    return ov_frac

def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight, crs, array):
    """
    create a raster file from a numpy array
    """

    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRaster.SetProjection(crs.ExportToWkt())
    outband.FlushCache()

def slope_stats_in_sheds(sheds_pth, dem_pth, cell_size=3.2809045, stats='mean median std count min max'):
    """
    Given a shapefile and a DEM raster, calculate the slope statistics in each zone.
    Return: array of dicts with stats for each zone
    """
    #open the dem raster, convert to np array
    raster = gdal.Open(dem_pth)
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = raster.GetGeoTransform()
    band = raster.GetRasterBand(1) #bands start at one
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    dem_arr = band.ReadAsArray().astype(np.float)

    #calculate slope x and y compenent slopes
    g = np.gradient(dem_arr, cell_size)
    xg = g[0]
    yg = g[1]

    #calculate magnitudes of resultant vectors
    slopes = np.sqrt(xg**2 + yg**2)

    #write the slopes dem array to a raster file
    wd = os.path.dirname(dem_pth)
    dem_rast_fname = os.path.splitext(os.path.basename(dem_pth))[0]
    slope_rast_path = os.path.join(wd, dem_rast_fname + '_slope.tif')
    array2raster(slope_rast_path, (upper_left_x, upper_left_y), x_size, y_size, raster_srs, slopes)

    #calculate the zonal stats
    slope_stats = zonal_stats(sheds_pth, slope_rast_path, stats=stats, nodata=-999)

    return slope_stats

def spatial_overlays(df1, df2, how='intersection'):

    '''Compute overlay intersection of two
        GeoPandasDataFrames df1 and df2
    '''
    df1 = df1.copy()
    df2 = df2.copy()
    df1['geometry'] = df1.geometry.buffer(0)
    df2['geometry'] = df2.geometry.buffer(0)
    if how=='intersection':
        # Spatial Index to create intersections
        spatial_index = df2.sindex
        df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
        df1['histreg']=df1.bbox.apply(lambda x:list(spatial_index.intersection(x)))
        pairs = df1['histreg'].to_dict()
        nei = []
        for i,j in pairs.items():
            for k in j:
                nei.append([i,k])

        pairs = gpd.GeoDataFrame(nei, columns=['idx1','idx2'], crs=df1.crs)
        pairs = pairs.merge(df1, left_on='idx1', right_index=True)
        pairs = pairs.merge(df2, left_on='idx2', right_index=True, suffixes=['_1','_2'])
        pairs['Intersection'] = pairs.apply(lambda x: (x['geometry_1'].intersection(x['geometry_2'])).buffer(0), axis=1)
        pairs = gpd.GeoDataFrame(pairs, columns=pairs.columns, crs=df1.crs)
        cols = pairs.columns.tolist()
        cols.remove('geometry_1')
        cols.remove('geometry_2')
        cols.remove('histreg')
        cols.remove('bbox')
        cols.remove('Intersection')
        dfinter = pairs[cols+['Intersection']].copy()
        dfinter.rename(columns={'Intersection':'geometry'}, inplace=True)
        dfinter = gpd.GeoDataFrame(dfinter, columns=dfinter.columns, crs=pairs.crs)
        dfinter = dfinter.loc[dfinter.geometry.is_empty==False]
        return dfinter
    elif how=='difference':
        spatial_index = df2.sindex
        df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
        df1['histreg']=df1.bbox.apply(lambda x:list(spatial_index.intersection(x)))
        df1['new_g'] = df1.apply(lambda x: reduce(lambda x, y: x.difference(y).buffer(0), [x.geometry]+list(df2.iloc[x.histreg].geometry)) , axis=1)
        df1.geometry = df1.new_g
        df1 = df1.loc[df1.geometry.is_empty==False].copy()
        df1.drop(['bbox', 'histreg', new_g], axis=1, inplace=True)
        return df1

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


def max_depth_from_raster(row, dem_pth, dem_adjustment=-4.63):
    """
    return Max Depth for a df row passed with X & Y columns,
    for use in a SWMM5 Junctions table
    """
    if pd.isnull(row.X) or pd.isnull(row.Y):
        return None

    point = 'POINT({} {})'.format(row.X, row.Y)
    rim_elev = point_query(point, dem_pth)[0] + dem_adjustment
    invert = row.InvertElev
    max_depth = rim_elev - invert

    return max_depth
