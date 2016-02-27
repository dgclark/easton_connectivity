from __future__ import division

import numpy as np
import plotly.graph_objs as go


def generate_scatter_data(data_path, sample_interval, marker_size):
  data = np.loadtxt(data_path)
  data_sparse = data[::sample_interval, :]

  roi_values = np.unique(data[:, 3])
  min_roi = np.min(roi_values)
  max_roi = np.max(roi_values)
  diff = max_roi - min_roi
  roi_norm = (data_sparse[:, 3] - min_roi) / diff

  return [go.Scatter3d(z=data_sparse[:, 2],
                       x=data_sparse[:, 0],
                       y=data_sparse[:, 1],
                       mode='markers',
                       showlegend=True,
                       marker=dict(color=roi_norm,
                                   size=marker_size,
                                   colorscale='Jet')
                       )]
