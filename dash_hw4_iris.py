
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([

    html.H1("Iris Dataset Explorer"),

    html.Div([

        html.Div([
            html.Label("X-axis"),
            dcc.Dropdown(
                id='x-axis',
                options=[{'label': col, 'value': col} for col in df.columns[:-1]],
                value='sepal length (cm)'
            ),

            html.Br(),

            html.Label("Color by Species"),
            dcc.Checklist(
                id='color-by-species',
                options=[{'label': 'Yes', 'value': 'target'}],
                value=[]
            )
        ], style={'width': '48%', 'display': 'inline-block',
                  'verticalAlign': 'top'}),

        html.Div([
            html.Label("Y-axis"),
            dcc.Dropdown(
                id='y-axis',
                options=[{'label': col, 'value': col} for col in df.columns[:-1]],
                value='sepal width (cm)'
            ),

            html.Br(),

            html.Label("Marker Size"),
            dcc.Slider(
                id='marker-size',
                min=5,
                max=20,
                step=1,
                value=10,
                marks={5: '5', 10: '10', 15: '15', 20: '20'}
            )
        ], style={'width': '48%', 'display': 'inline-block',
                  'marginLeft': '4%', 'verticalAlign': 'top'})

    ]),

    html.Br(),

    dcc.Graph(id='scatter-plot', style={'width': '100%'}),
    dcc.Graph(id='histogram', style={'width': '100%'})

], style={'padding': '20px'})


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    Input('color-by-species', 'value'),
    Input('marker-size', 'value')
)
def update_scatter(x, y, color_species, size):

    color = 'target' if 'target' in color_species else None

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


@app.callback(
    Output('histogram', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    Input('color-by-species', 'value')
)
def update_histogram(x, y, color_species):

    color = 'target' if 'target' in color_species else None

    fig = px.histogram(
        df,
        x=y,
        color=color,
        barmode='group',
        title=f'Histogram of {y}'
    )

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run(debug=True)
