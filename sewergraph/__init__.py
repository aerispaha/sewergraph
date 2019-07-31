################################################################################
# Module: __init__.py
# Description: Tool for graph calculations on drainage networks
# License: MIT, see full license in LICENSE.txt
# Web: https://github.com/aerispaha/sewergraph
################################################################################

from .core import *
from .helpers import *
from .area_calcs import *
from .resolve_data import *
from sewergraph.save_load import graph_from_shp, gdf_from_graph, graph_from_gdf

VERSION_INFO = (0, 1, 2)
__version__ = '.'.join(map(str, VERSION_INFO))
__author__ = 'Adam Erispaha'
__copyright__ = 'Copyright (c) 2017 Adam Erispaha'
__licence__ = ''
