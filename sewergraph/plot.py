from plotly.graph_objs import Figure, Scatter, Bar, Layout
import numpy as np
from . import helpers

def plot_profile(G, path):

    l = 0
    xs = []
    inverts = []
    heights = []
    rims = []
    fids = []

    for u,v,d in G.edges(data=True, nbunch=path):

        node = G.nodes[u]
        sewer = G[u][v]
        inverts.append(node['invert'])
        h = max(sewer['diameter'], sewer['height']) / 12.0
        heights.append(h + node['invert'])
        rim_el = node.get('ELEVATION_', None)
        rims.append(rim_el)
        xs.append(l)
        fids.append(sewer['facilityid'])
        l += sewer['length']


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


def cdf_go(net, parameter='capacity_fraction', cumu_param = 'length',
           normal=1, units='', df=None, name =None):
    """
    create a Plotly graph object of a cumulative distribution function across
    the edges
    """
    #sort and accumulate length
    if df is None:
        df = net.conduits()
    if name is None:
        name = net.name

    df = df.sort_values(by=parameter, ascending=True)
    df['cumulated_param'] = df[cumu_param].cumsum()
    df['cumulated_param_frac'] = df.cumulated_param / df[cumu_param].sum()
    df['cumu_converted'] = df.cumulated_param / normal
    text = ['<br>'.join(x) + units
            for x in zip(df.LABEL.astype(str).tolist(),
                         df.cumu_converted.round(1).astype(str).tolist())]
    graph_obj = Scatter(
        x = df[parameter],
        y = df.cumulated_param_frac,
        text = text,
        name = name,
    )

    return graph_obj


def cumulative_distribution(df, name = 'Capacity Surplus',
                            title='Sewer Capacity Surplus By Length',
                            parameter='capacity_fraction',
                            cumu_param = 'length'
                            ):


    #sort and accumulate length
    cap = df.sort_values(by=parameter, ascending=True)
    cap['cumulated_param'] = cap[cumu_param].cumsum()
    cap['cumulated_param_frac'] = cap.cumulated_param / cap[cumu_param].sum()
    cap['cumu_miles'] = cap.cumulated_param #/ 5280.0

    cum_cap = Scatter(
        x = cap[parameter],
        y = cap.cumulated_param_frac,
        text = ['<br>'.join(x) + 'mi' for x in zip(cap.LABEL.astype(str).tolist(), cap.cumu_miles.round(1).astype(str).tolist())] ,
        name = name,
    )
    layout = Layout(
        xaxis=dict(
            #range=[-100, 100],
            range=[0, 2.5],
            title=parameter,
        ),
        yaxis=dict(
            title=cumu_param + ' fraction',
        ),
        title=title,
        hovermode='closest'
    )

    data = [cum_cap]
    fig = Figure(data=data, layout = layout)

    return fig, data, layout
