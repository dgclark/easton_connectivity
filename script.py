from __future__ import division

import numpy as np
import pandas as pd


def main_part(num_perms=3, cut_off=.9, output_file=None):
  data = pd.read_csv("ROI_matrix.txt", sep="\t")

  is_normal = np.logical_or(data.dx=='nc', data.dx=='aami')
  normals = data[is_normal]

  normal_rois = normals.loc[:, normals.columns[3:]]

  maxes = sorted_permutations(normal_rois.get_values(), num_perms)

  normal_rois_cov = np.corrcoef(normal_rois.T)
  normal_rois_cov = pd.DataFrame(normal_rois_cov,
                                 index=normal_rois.columns,
                                 columns=normal_rois.columns)

  valid_connections = connections_above(maxes, cut_off, normal_rois_cov)

  if output_file is not None:
    valid_connections.to_csv(output_file)

  return valid_connections


def connections_above(sorted_distribution, cut_off, connection_df):
  """
  :param sorted_distribution:
  :param cut_off:
  :param connection_df:
  :return connection_mask:
  >>> import pandas as pd
  >>> import numpy as np
  >>> rois = ['a', 'b']
  >>> connection_df = pd.DataFrame(np.array([[3, 4], [4, 3]]), index=rois, columns=rois)
  >>> sorted_distribution = [2, 3, 4]
  >>> cut_off = .7
  >>> expected = pd.DataFrame(np.array([[False, True], [True, False]]), index=rois, columns=rois)
  >>> actual = connections_above(sorted_distribution, cut_off, connection_df)
  >>> assert(np.all(expected == actual))
  """
  return connection_df > cutoff_value(sorted_distribution, cut_off)


def cutoff_value(sorted_arr, cut_off):
  """
  :param arr:
  :param cut_off:
  :return:
  >>> import numpy as np
  >>> sorted_arr = np.array([2, 3, 4, 5, 6])
  >>> num_above = lambda cut_off: np.sum(sorted_arr > cutoff_value(sorted_arr, cut_off))
  >>> assert(num_above(.2) == 4)
  >>> assert(num_above(.51) == 2)
  >>> assert(cutoff_value(sorted_arr, .2) == 2)
  >>> assert(cutoff_value(sorted_arr, .1) == 1)
  >>> assert(num_above(1) == 0)
  """
  total = len(sorted_arr)
  cut_off = cut_off if cut_off != .5 else .51 #numpy rounds .5 down, want up
  index = int(np.round(total * cut_off)) - 1
  return sorted_arr[index] if index > -1 else sorted_arr[0] - 1


def sorted_permutations(mat, num_perms):
  """
  :param mat:
  :param num_perms:
  :return corr_maxes:
  >>> x = np.array([[1, 2, 3], [2, 3, 1]]).T
  >>> num_perms = 15 # 6 possible combos but to handle repeats
  >>> corr_maxes = set(sorted_permutations(x, num_perms))
  >>> expected = set((-.5, .5, 1.))
  >>> assert(corr_maxes == expected)
  """
  maxes = np.zeros(num_perms)
  num_rows, num_cols = mat.shape

  for perm in range(num_perms):
    row_indexes = range(num_rows)
    np.random.shuffle(row_indexes)
    for col_ix in range(num_cols):
      corrs = corr_with_shuffle(mat, row_indexes, col_ix)
      maxes[perm] = np.max(corrs)

  return np.sort(maxes)


def corr_with_shuffle(mat, shuffle_indexes, col_index):
  """
  :param mat:
  :param shuffle_indexes:
  :param col_index:
  :return corrs:
  >>> import numpy as np
  >>> x = np.array([[1, 2, 3], [2, 3, 1]]).T
  >>> shuffle_indexes = [2, 1, 0]
  >>> col_index = 0
  >>> expected = np.array([-1, .5])
  >>> y = corr_with_shuffle(x, shuffle_indexes, col_index)
  >>> assert(np.all(y == expected))
  """
  col_shuffled = mat[shuffle_indexes, col_index]

  return np.corrcoef(mat.T, col_shuffled)[-1,:-1]


if __name__ == "__main__":
  import doctest
  doctest.testmod()