from plotly.graph_objs import Figure, Scatter, Bar, Layout
import numpy as np

def plot_profile(G, path):

    l = 0
    xs = []
    inverts = []
    heights = []
    rims = []
    fids = []

    for u,v,d in G.edges_iter(data=True, nbunch=path):

        node = G.node[u]
        sewer = G[u][v]
        inverts.append(node['invert'])
        h = max(sewer['Diameter'], sewer['Height']) / 12.0
        heights.append(h + node['invert'])
        rim_el = node.get('ELEVATION_', None)
        rims.append(rim_el)
        xs.append(l)
        fids.append(sewer['FACILITYID'])
        l += sewer['Shape_Leng']


    inv_go = Scatter(
        x = xs,
        y = inverts,
        name='invert',
        text = fids,
    )
    h_go = Scatter(
        x = xs,
        y = heights,
        name='heights'
    )

    rims_go = Scatter(
        x = xs,
        y = [None if r == 0 else r for r in rims],
        name='rims',
        connectgaps= True,
    )

    return Figure(data=[inv_go, h_go, rims_go])


def capacity_peak_comparison_plt(df, name ='Sewers', title='Peak Flow vs Sewer Capacity'):

    df['cap_frac'] = df.peakQ / df.capacity
    df['phs'] = df.capacity / df.upstream_area_ac
    df = df.sort_values(by='peakQ')
    maxv = max(df.peakQ.max(), df.capacity.max())

    scatter = Scatter(
        x = df.peakQ.round(3).tolist(),
        y = df.capacity.round(3).tolist(),
        text = df.LABEL.tolist(),
        mode='markers',
        name = name,
    )
    one_one = Scatter(
        x = np.linspace(0.001, maxv),
        y = np.linspace(0.001, maxv),
        text = df.LABEL.tolist(),
        name = 'One to One',
    )
    # lims = lims.sort_values(by='phs_rate', ascending=False)

    layout = Layout(
        xaxis=dict(
            type='log',
            autorange=True,
            title='Peak Flow (cfs)',
            range=[0.1, maxv],
        ),
        yaxis=dict(
            type='log',
            autorange=True,
            title='Capacity (cfs)',
        ),
        title=title,
        hovermode='closest'
    )
    data = [scatter, one_one]
    fig = Figure(data = data, layout=layout)


    return fig, data, layout


def cumulative_distribution(df, name = 'Capacity Surplus',
                            title='Sewer Capacity Surplus By Length'):
    #capacity fraction cumulative distribution
    df['capacity_frac'] = df.peakQ / df.capacity
    df['capacity_deficit_frac'] = (df.capacity - df.peakQ) #/ df.capacity
    cols = ['capacity_frac', 'capacity', 'peakQ',  'capacity_deficit_frac',
            'upstream_area_ac', 'Shape_Leng', 'LABEL', 'Year_Insta', 'FACILITYID']
    cap = df[cols].set_index('FACILITYID')

    #sort and accumulate length
    #cap = cap.sort_values(by='capacity_deficit_frac', ascending=True)
    cap = cap.sort_values(by='capacity_frac', ascending=True)
    cap['cumu_length'] = cap.Shape_Leng.cumsum()
    cap['cumu_length_frac'] = cap.cumu_length / cap.Shape_Leng.sum()
    cap['cumu_miles'] = cap.cumu_length / 5280

    # peak = cap[:]
    # peak = peak.sort_values(by='peakQ')
    # peak['cumu_length'] = peak.Shape_Leng.cumsum()
    # peak['cumu_length_frac'] = peak.cumu_length / peak.Shape_Leng.sum()
    # peak['cumu_miles'] = peak.cumu_length / 5280


    cum_cap = Scatter(
        x = cap.capacity_frac.tolist(),
        y = cap.cumu_length_frac.tolist(),
        text = ['<br>'.join(x) + 'mi' for x in zip(cap.LABEL.astype(str).tolist(), cap.cumu_miles.round().astype(str).tolist())] ,
        name = name,
    )
    layout = Layout(
        xaxis=dict(
            #range=[-100, 100],
            range=[0, 2.5],
            title='Capacity Fraction',
        ),
        yaxis=dict(
            title='Fraction of Sewer (Length)',
        ),
        title=title,
        hovermode='closest'
    )

    data = [cum_cap]
    fig = Figure(data=data, layout = layout)

    return fig, data, layout
