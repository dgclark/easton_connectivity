from __future__ import division

import numpy as np

def main_part(output_file='labels.txt'):
    adj_mat = np.loadtxt('roi_adjacency.txt', delimiter=',')
    labels = apply_labels(adj_mat)

    if output_file is not None:
        np.savetxt(output_file, labels, delimiter=',', fmt='%d')

    return labels


def apply_labels(adj_mat):
    num_rois = adj_mat.shape[1]
    labels = np.zeros(num_rois, dtype=int)
    for roi_ix in range(num_rois):
        update_labels(roi_ix, adj_mat, labels)
    return labels


def update_labels(roi_ix, adj_mat, labels):
    """
    >>> import numpy as np
    >>> roi_ix = 0
    >>> adj_mat = np.zeros((4, 4))
    >>> neighbors = [(0, 3), (0, 1)]
    >>> for n in neighbors:
    ...  adj_mat[n] = 1
    ...  adj_mat[n[::-1]] = 1
    >>> labels = np.zeros(4)
    >>> update_labels(0, adj_mat, labels)
    >>> expected = np.array([0, 1, 0, 2])
    >>> assert(np.all(expected == labels))
    """
    roi_neighbors = adj_mat[:, roi_ix] > 0
    for n in np.flatnonzero(roi_neighbors):
        n_neighbors = adj_mat[:, n] > 0
        all_neighbors = np.logical_or(roi_neighbors, n_neighbors)
        label = get_new_label(labels[all_neighbors])
        labels[n] = label


def get_new_label(labels):
    """
    >>> import numpy as np
    >>> labels = np.array([0, 1, 2, 3])
    >>> expected = 4
    >>> assert(expected == get_new_label(labels))
    >>> labels = np.array([2, 0, 3, 10])
    >>> expected = 1
    >>> assert(expected == get_new_label(labels))
    """
    labels = set(labels)

    def smallest_new_label(curr_test=1):
        return curr_test if curr_test not in labels else smallest_new_label(curr_test+1)

    return smallest_new_label()


if __name__ == "__main__":
    import doctest
    doctest.testmod()