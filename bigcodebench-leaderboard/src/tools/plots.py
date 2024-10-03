import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def plot_elo_mle(df):
    fig = px.scatter(df, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus",
                    #  title="Bootstrap of Elo MLE Estimates (BigCodeBench-Complete)"
                     )
    fig.update_layout(xaxis_title="Model", 
                      yaxis_title="Rating",
                      autosize=True,
                    #   width=1300,
                    #   height=900,
                      )
    return fig


def plot_solve_rate(df, task, rows=30, cols=38):
    keys = df["task_id"]
    values = df["solve_rate"]
    
    values = np.array(values, dtype=float)  # Ensure values are floats

    # Extract numerical IDs and sort by them
    ids = [int(key.split('/')[-1]) for key in keys]
    sorted_indices = np.argsort(ids)
    keys = np.array(keys)[sorted_indices]
    values = values[sorted_indices]
    
    n = len(values)
    pad_width = rows * cols - n
    
    # Create a masked array
    masked_values = np.ma.array(np.full(rows * cols, np.nan), mask=True)
    masked_values[:n] = values
    masked_values.mask[:n] = False
    masked_values = masked_values.reshape((rows, cols))

    keys_padded = np.pad(keys, (0, pad_width), 'constant', constant_values='')
    keys_reshaped = keys_padded.reshape((rows, cols))

    hover_text = np.empty_like(masked_values, dtype=object)
    for i in range(rows):
        for j in range(cols):
            if not masked_values.mask[i, j]:
                hover_text[i, j] = f"{keys_reshaped[i, j]}<br>Solve Rate: {masked_values[i, j]:.2f}"
            else:
                hover_text[i, j] = "NaN"

    upper_solve_rate = round(np.count_nonzero(values) / n * 100, 2)
    
    fig = go.Figure(data=go.Heatmap(
        z=masked_values,
        text=hover_text,
        hoverinfo='text',
        colorscale='teal',
        zmin=0,
        zmax=100
    ))

    fig.update_layout(
        title=f'BigCodeBench-{task}<br><i>Lowest Upper Limit: {upper_solve_rate}%</i>',
        xaxis_nticks=cols,
        yaxis_nticks=rows,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        autosize=True,
    )
    
    return fig