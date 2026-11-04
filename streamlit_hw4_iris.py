
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris


iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

def main():
    st.title("Iris Dataset Explorer")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox('X-axis', options=list(df.columns[:-1]), index=0)
        color_by_species = st.checkbox('Color by Species', value=False)

    with col2:
        y_axis = st.selectbox('Y-axis', options=list(df.columns[:-1]), index=1)
        marker_size = st.slider('Marker Size', min_value=5,
                                max_value=20, value=10, step=1)

    color = 'target' if color_by_species else None

    fig = px.scatter(df, x=x_axis, y=y_axis, color=color,
                     title=f'Iris Dataset: {x_axis} vs {y_axis}')

    fig.update_traces(marker=dict(size=marker_size))

    st.plotly_chart(fig, width="stretch")

    fig_hist = px.histogram(df, x=y_axis, color=color,
                            barmode='group',
                            title=f'Histogram of {y_axis}')

    st.plotly_chart(fig_hist, width="stretch")

if __name__ == "__main__":
    main()
