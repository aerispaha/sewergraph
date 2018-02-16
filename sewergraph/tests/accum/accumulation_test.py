import networkx as nx
import sewergraph as sg

def test_downstream_accum():

    H = nx.DiGraph()
    H.add_edges_from([(5,4), (4,3), (6,3), (3,2), (2,1)])
    H.node[5]['local_area'] = 1
    H.node[4]['local_area'] = 1.75
    H.node[3]['local_area'] = 1

    H[6][3]['local_area'] = 0.25

    H = sg.accumulate_downstream(H)
    assert H.node[1]['cumulative_local_area'] == 4.0

    #add a flow split
    H.add_edges_from([(1,0), (1, 'A')])
    H[1][0]['flow_split_frac'] = 0.25
    H[1]['A']['flow_split_frac'] = 0.75
    H = sg.accumulate_downstream(H, split_attr='flow_split_frac')

    assert H.node[0]['cumulative_local_area'] == 1.0
    assert H.node['A']['cumulative_local_area'] == 3.0


def test_identify_outfalls():
    H = nx.DiGraph()
    H.add_edges_from([(99,3), (3,2), (2,'outfall1'), (2,'a'), ('a','b'),
                      ('b', 'outfall2'), ('b','outfall3')])

    H1 = sg.identify_outfalls(H)

    assert (G4.node[2]['outfalls'] == ['outfall3', 'outfall2', 'outfall1'])
    assert (G4.node['b']['outfalls'] == ['outfall3', 'outfall2'])

def test_houboutdat():
    assert 5==5
