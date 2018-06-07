# Graph Tools for Sewers
Building upon Networkx, this package seeks to provide tools for analysis
and manipulation of sewer network data. This package is under active
development in the osmnx branch, where the original code being reworked
to emulate the very, very nice [osmnx](https://github.com/gboeing/osmnx)
package. Be advised that things will change - the master branch will be
merged with osmnx after [Milestone 1](https://github.com/aerispaha/sewergraph/milestone/1)
for v0.1.0.

# Goals
What are we hoping to accomplish here.
- Traverse sewer networks up/downstream
- Interface for accumulation calculations
- Data gap handling (within reason)
- import/export open-source spatial data formats (shapefiles, geojson)
- some visualization
- basic hydrologic/hydraulic calculations (Rational, Mannings)


### Example
```python

G = nx.read_shp(shapefile)

#clean up the network (rm unecessary DataConv fields, isolated nodes)
G = clean_network_data(G)
G = round_shapefile_node_keys(G)
G = nx.convert_node_labels_to_integers(G, label_attribute='coords')
G = resolve_geom_gaps(G)

#perform capacity calcs
G = hhcalcs_on_network(G)

#id flow split sewers and calculate split fractions
G = analyze_flow_splits(G)

if boundary_conditions is not None:
    add_boundary_conditions(G, boundary_conditions)
self.boundary_conditions = boundary_conditions

#accumulate drainage areas
G = accumulate_area(G, cumu_attr_name='cumulative_area')
G = propogate_weighted_C(G, gsi_capture)
G = resolve_slope_gaps(G)
G = hhcalcs_on_network(G)

#accumulating travel times
G = accumulate_travel_time(G)
```
