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
