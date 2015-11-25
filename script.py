import numpy as np
import pandas as pd


def main_part():
  data = pd.read_csv("ROI_matrix.txt", sep="\t")

  is_normal = np.logical_or(data.dx=='nc', data.dx=='aami')
  normals = data[is_normal]

  normal_rois = normals.loc[:, normals.columns[3:]].get_values()
  normal_rois_cov = np.corrcoef(normal_rois.T)

  num_perms = 3
  maxes = sorted_permutations(normal_rois, num_perms)

  return maxes


def sorted_permutations(mat, num_perms):
  """
  :param mat:
  :param num_perms:
  :return corr_maxes:
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
  num_cols = mat.shape[1]
  corrs = np.zeros(num_cols)

  for col_ix in range(num_cols):
    corrs[col_ix] = np.corrcoef(mat[:, col_ix], col_shuffled)[0][1]

  return corrs


if __name__ == "__main__":
  import doctest
  doctest.testmod()