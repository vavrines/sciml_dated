import numpy as np
import plotly

data = np.load("pdf.npz")
#data = np.load("/home2/vavrines/Coding/sciml/bgknet/cavity/pdf.npz")

u = data['u']
v = data['v']
x = data['x']
y = data['y']
qsx = data['qsx']
qsy = data['qsy']
qbx = data['qbx']
qby = data['qby']

go = plotly.graph_objects
fig = go.Figure()
fig.add_trace(
    go.Volume(
        x=u.ravel(),
        y=v.ravel(),
        z=x.ravel(),
        value=qsx.ravel(),
        opacity=0.1,
        surface_count=20,
        colorscale="PiYG",
    ),
)
fig.update_layout(
    scene=dict(xaxis_title="u", yaxis_title="v", zaxis_title="x"),
)
# fig.show()
fig.write_image("cavity_qsx.pdf")

go = plotly.graph_objects
fig = go.Figure()
fig.add_trace(
    go.Volume(
        x=u.ravel(),
        y=v.ravel(),
        z=x.ravel(),
        value=qbx.ravel(),
        opacity=0.1,
        surface_count=20,
        colorscale="PiYG",
    ),
)
fig.update_layout(
    scene=dict(xaxis_title="u", yaxis_title="v", zaxis_title="x"),
)
# fig.show()
fig.write_image("cavity_qbx.pdf")

go = plotly.graph_objects
fig = go.Figure()
fig.add_trace(
    go.Volume(
        x=u.ravel(),
        y=v.ravel(),
        z=y.ravel(),
        value=qsy.ravel(),
        opacity=0.1,
        surface_count=20,
        colorscale="PiYG",
    ),
)
fig.update_layout(
    scene=dict(xaxis_title="u", yaxis_title="v", zaxis_title="y"),
)
# fig.show()
fig.write_image("cavity_qsy.pdf")

go = plotly.graph_objects
fig = go.Figure()
fig.add_trace(
    go.Volume(
        x=u.ravel(),
        y=v.ravel(),
        z=y.ravel(),
        value=qby.ravel(),
        opacity=0.1,
        surface_count=20,
        colorscale="PiYG",
    ),
)
fig.update_layout(
    scene=dict(xaxis_title="u", yaxis_title="v", zaxis_title="y"),
)
# fig.show()
fig.write_image("cavity_qby.pdf")
