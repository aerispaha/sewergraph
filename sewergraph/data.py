nodes = nx.ancestors(G, 1990) | nx.ancestors(G, 185) | nx.ancestors(G, 2558) | nx.ancestors(G, 3092)
nodes = nodes | nx.ancestors(G, 373) | nx.ancestors(G, 4119) | nx.ancestors(G, 1447)

fids = [G.node[n]['facilityid'] for n in nodes if 'facilityid' in G.node[n]]

boundary = {fid:{'runoff_coefficient':0.5} for fid in fids}

gsi = {212:{'fraction':0.10, 'C':0.35}}
