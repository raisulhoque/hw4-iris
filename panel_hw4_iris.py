import panel as pn
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

pn.extension('plotly')

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

x_axis = pn.widgets.Select(
    name='X-axis',
    options=list(df.columns[:-1]),
    value=list(df.columns[:-1])[0]
)

y_axis = pn.widgets.Select(
    name='Y-axis',
    options=list(df.columns[:-1]),
    value=list(df.columns[:-1])[1]
)

color_by_species = pn.widgets.Checkbox(
    name='Color by Species',
    value=False
)

marker_size = pn.widgets.IntSlider(
    name='Marker Size',
    start=5,
    end=20,
    step=1,
    value=10
)


def update_scatter(x, y, color_species, size):

    color = 'target' if color_species else None

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        title=f'Iris Dataset: {x} vs {y}'
    )

    fig.update_traces(marker=dict(size=size))
    fig.update_layout(transition_duration=500)

    return fig


def update_histogram(x, y, color_species):

    color = 'target' if color_species else None

    fig = px.histogram(
        df,
        x=y,
        color=color,
        barmode='group',
        title=f'Histogram of {y}'
    )

    fig.update_layout(transition_duration=500)

    return fig


scatter_plot = pn.pane.Plotly(
    pn.bind(
        update_scatter,
        x_axis,
        y_axis,
        color_by_species,
        marker_size
    ),
    sizing_mode='stretch_width'
)

histogram = pn.pane.Plotly(
    pn.bind(
        update_histogram,
        x_axis,
        y_axis,
        color_by_species
    ),
    sizing_mode='stretch_width'
)

layout = pn.Column(
    "# Iris Dataset Explorer",

    pn.Row(
        pn.Column(
            x_axis,
            color_by_species,
            sizing_mode='stretch_width'
        ),

        pn.Column(
            y_axis,
            marker_size,
            sizing_mode='stretch_width'
        ),
        sizing_mode='stretch_width'
    ),

    scatter_plot,
    histogram,

    sizing_mode='stretch_width'
)

layout.servable()
