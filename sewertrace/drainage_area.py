
# coding: utf-8

# In[105]:

import scipy
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely.geometry
import shapely.ops
from shapely.wkt import loads
from descartes import PolygonPatch
from matplotlib import pyplot
from sewertrace import trace, helpers
import pyproj
from functools import partial
from geojson import Polygon, Feature, FeatureCollection, LineString
import json


# In[124]:

reload(trace)
#Read the shapefile data into a graph
data_dir = r'/Users/adam/Desktop/SewerPlanning/OldMaps/ss5749_split2'
G = nx.read_shp(data_dir)
G = nx.convert_node_labels_to_integers(G, label_attribute='coords')
for u,v,d in G.edges_iter(data=True):
    if 'Wkb' in d:
        del d['Wkb']


# In[113]:

# nx.topological_sort(G)
# G = G.subgraph(nx.descendants(G, source=401))


# In[125]:

#create shapely objects of sewer polylines and their midpoints
sewer_lines = [loads(d['Wkt']) for u,v,d in G.edges_iter(data=True)]
centroids = [(line.centroid.x, line.centroid.y) for line in sewer_lines]

sewers = shapely.geometry.MultiLineString(sewer_lines)
sewers


# In[126]:

#create a buffer around the study area
study_area = sewers.convex_hull.buffer(distance=100, cap_style=3, join_style=3)
border_pts = [xy for xy in study_area.boundary.coords]
study_area.boundary


# In[127]:

vor = Voronoi(centroids+border_pts)
voronoi_plot_2d(vor, show_vertices=False)


# In[129]:

#create shapely polygons of drainage areas
drainage_bounds = [
    shapely.geometry.LineString(vor.vertices[line])
    for line in vor.ridge_vertices
    if -1 not in line
]

da_list = [
    da.intersection(study_area)
    for da in shapely.ops.polygonize(drainage_bounds)
]


#convert to wgs projection system
pa_plane = pyproj.Proj(init='epsg:2272', preserve_units=True)
wgs = pyproj.Proj(proj='longlat', datum='WGS84', ellps='WGS84') #google maps, etc
project = partial(pyproj.transform, pa_plane, wgs)
da_list_wgs = [shapely.ops.transform(project, da) for da in da_list]

#create a multipolygon
das = shapely.geometry.MultiPolygon(da_list_wgs)
das


# In[131]:

reload(helpers)
#create a geojson FeatureCollection
da_feats = [Feature(geometry=geom, properties=dict(area=geom.area)) for geom in das]
da_geo = FeatureCollection(da_feats)

#sewer geojson
# sewer_geo = helpers.write_geojson(G)

lyrs = dict(
    conduits = helpers.write_geojson(G),
    sheds = da_geo
)
filename = r'/Users/adam/Desktop/cool.html'
helpers.create_html_map(lyrs, filename, G,'vornoi.html')


# In[ ]:




# In[33]:

p = Polygon(coordinates=list(geom.boundary.coords))
print Feature(geometry=geom)


# In[ ]:
