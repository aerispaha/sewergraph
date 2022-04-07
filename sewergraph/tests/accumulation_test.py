import networkx as nx
import sewergraph as sg
import os

import pytest

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')


def test_downstream_accum():

    H = nx.MultiDiGraph()
    H.add_edges_from([(5,4), (4,3), (6,3), (3,2), (2,1)])
    H.nodes[5]['local_area'] = 1
    H.nodes[4]['local_area'] = 1.75
    H.nodes[3]['local_area'] = 1

    H[6][3][0]['local_area'] = 0.25

    H = sg.accumulate_downstream(H)
    assert H.nodes[1]['cumulative_local_area'] == 4.0

    #add a flow split
    H.add_edges_from([(1,0), (1, 'A')])
    H[1][0][0]['flow_split_frac'] = 0.25
    H[1]['A'][0]['flow_split_frac'] = 0.75
    H = sg.accumulate_downstream(H, split_attr='flow_split_frac')

    assert H.nodes[0]['cumulative_local_area'] == 1.0
    assert H.nodes['A']['cumulative_local_area'] == 3.0


# def test_identify_outfalls():
#     H = nx.DiGraph()
#     H.add_edges_from([(99,3), (3,2), (2,'outfall1'), (2,'a'), ('a','b'),
#                       ('b', 'outfall2'), ('b','outfall3')])
#
#     H1 = sg.identify_outfalls(H)
#
#     assert (H1.nodes[2]['outfalls'] == ['outfall3', 'outfall2', 'outfall1'])
#     assert (H1.nodes['b']['outfalls'] == ['outfall3', 'outfall2'])


def test_relative_outfall_contribution():
    H = nx.MultiDiGraph()
    H.add_edges_from([('A','i'), ('B','i'), ('C','j'), ('D','k'),
                      ('i', 'j'), ('j','k'),  ('k','OF2')])

    H.nodes['A']['local_area'] = 1.0
    H.nodes['B']['local_area'] = 2.0
    H.nodes['C']['local_area'] = 1.0
    H.nodes['D']['local_area'] = 1.0

    #flow splits
    H.add_edges_from([('j','j1'), ('j1', 'OF1')])
    H['j']['j1'][0]['flow_split_frac'] = 0.25
    H['j']['k'][0]['flow_split_frac'] = 0.75

    H = sg.accumulate_downstream(H, accum_attr='local_area',
                                 cumu_attr_name='cumulative_area')
    H = sg.assign_inflow_ratio(H, inflow_attr='cumulative_area')
    H = sg.relative_outfall_contribution(H)

    assert(H.nodes['B']['outfall_contrib'] == {'OF2': 0.4, 'OF1': 0.5})
    assert(H.nodes['j']['outfall_contrib'] == {'OF2': 0.8, 'OF1': 1.0})
    assert(H.nodes['k']['outfall_contrib'] == {'OF2': 1.0})


@pytest.mark.skip(reason="graph_from_shp is depreciated and doesn't support MultiDiGraph")
def test_graph_from_shp():

    #read shapefile into DiGraph
    shp_pth = os.path.join(DATA_DIR, 'sample_sewer_network_1.shp')
    G = sg.graph_from_shp(shp_pth)

    #basic shapefile read tests
    edge = G[19][59]
    assert (edge['facilityid'] == 'A58064DF')
    assert (round(edge['local_area'], 3) == 119043.525)

    #accumulate drainage area
    G = sg.core.accumulate_downstream(G, 'local_area', 'total_area')
    edge = G[19][59]
    assert (round(edge['total_area'], 3) == 4233504.422)
