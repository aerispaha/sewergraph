# Network Analysis for Sewers
Building upon Networkx, this package provides tools for analysis and manipulation
of sewer network data.

## Goals
Provide graph functions to tackle analytical problems typical in sewer
collections systems:  
- traverse sewer networks up/downstream
- accumulation calculations
- downstream choke-point analysis
- data gap handling (within reason, folks)
- design capacity analysis

## Installation
Since sewergraph is inspired by [osmnx](https://github.com/gboeing/osmnx), this
package depends on [GeoPandas](https://github.com/geopandas/geopandas), [Networkx](https://github.com/networkx/networkx), and a handful of other
packages. It's recommended to install GeoPandas with conda first, then install
sewergraph via pip:

```bash
$ conda install geopandas
$ pip install sewergraph
```
If you have ArcMap installed, be sure that the GeoPandas installation doesn't conflict with `arcpy`. To avoid risks, install sewergraph in a `conda` environment:

```bash
$ conda create --name myenv
$ activate myenv #enter the new environment
$ conda install geopandas
$ pip install sewergraph
```


### Examples
Create a Networkx DiGraph with a shapefile of a sewer network.
```python
import sewergraph as sg

#read shapefile into DiGraph
shapefile_path = r'path/to/sewers.shp'
G = sg.graph_from_shp(shapefile_path)
```

Attributes of each sewer segment are stored as edge data. Geometry is parse and stored in the `geometry` attribute along with whatever other fields exist in the shapefile.
```python
#sewer connecting node 0 to node 1
print(G[0][1])
```
```bash
{
  'OBJECTID': 115081,
  'STREET': 'ADAINVILLE DR',
  'ShpName': 'sample_sewer_network_1',
  'diameter': 8,
  'facilityid': 'BCE7B25E',
  'geometry': <shapely.geometry.linestring.LineString at 0x12a6caf0>,
  'height': 0,
  'length': 164.758,
  'local_area': 39449.474,
  'material': 'VCP',
  'pipeshape': 'CIR',
  'slope': 0.01,
  'width': 0
}
```

Calculate the total drainage area at each sewer by accumulating `local_area` from the top to bottom of the network (i.e. sewershed).

```python
#accumulate drainage area
G = sg.accumulate_downstream(G, 'local_area', 'total_area')

#convert to GeoDataFrame and sort the sewers by total_area
sewers = sg.gdf_from_graph(G)
sewers = sewers.sort_values(by = 'total_area', ascending=False)
sewers.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_area</th>
      <th>OBJECTID</th>
      <th>facilityid</th>
      <th>pipeshape</th>
      <th>diameter</th>
      <th>height</th>
      <th>width</th>
      <th>length</th>
      <th>slope</th>
      <th>material</th>
      <th>STREET</th>
      <th>local_area</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>4,233,504</td>
      <td>112545</td>
      <td>A58064DF</td>
      <td>BOX</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
      <td>327.279370</td>
      <td>0.0075</td>
      <td>RCP</td>
      <td>None</td>
      <td>119043.524941</td>
      <td>LINESTRING (6558821.45028765 2032961.24586616,...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4,114,461</td>
      <td>112546</td>
      <td>5890D18F</td>
      <td>BOX</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
      <td>318.081402</td>
      <td>0.0100</td>
      <td>RCP</td>
      <td>None</td>
      <td>171961.403740</td>
      <td>LINESTRING (6558826.08945222 2032643.19829701,...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3,942,499</td>
      <td>112563</td>
      <td>12FF7372</td>
      <td>BOX</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
      <td>131.352534</td>
      <td>0.0100</td>
      <td>RCP</td>
      <td>None</td>
      <td>16557.605522</td>
      <td>LINESTRING (6558821.78250872 2032511.9163921, ...</td>
    </tr>

  </tbody>
</table>
</div>

More functions are provided for calculating basic hydraulic capacity, outfall loading, flow splits, travel time, and identifying downstream constrictions from every point of the network.  

```python
#perform basic sewer capacity calculations (full flow Mannings capacity)
G = hhcalcs_on_network(G)

#id flow split sewers and calculate split fractions
G = analyze_flow_splits(G)

#accumulating travel times
G = accumulate_travel_time(G)
```

## Running Tests
Test are located in the sewergraph > tests directory and are run with `pytest`.  
