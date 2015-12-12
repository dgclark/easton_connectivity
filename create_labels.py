from __future__ import division

import numpy as np

def main_part(output_file='labels.txt'):
    adj_mat = np.loadtxt('roi_adjacency.txt', delimiter=',')

    is_valid = verify_valid_adj_mat(adj_mat)
    assert is_valid[0], 'adj matrix not valid, roi %s' % is_valid[1] + " " + str(is_valid[2])

    labels = apply_labels(adj_mat)

    is_valid = verify_valid_labels(adj_mat, labels)

    assert is_valid[0], "roi %s shares label with neighbor(s): " % is_valid[1] + " " + str(is_valid[2]) \
                        + " label: " + str(labels[is_valid[1]])

    if output_file is not None:
        np.savetxt(output_file, labels, delimiter=',', fmt='%d')

    return labels


def apply_labels(adj_mat):
    """
    >>> import numpy as np
    >>> adj_mat = np.zeros((4, 4))
    >>> neighbors = [(0, 3), (0, 1)]
    >>> for n in neighbors:
    ...  adj_mat[n] = 1
    ...  adj_mat[n[::-1]] = 1
    >>> labels = apply_labels(adj_mat)
    >>> expected = np.array([1, 2, 1, 2])
    >>> assert(np.all(expected==labels))
    """
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
    >>> expected = np.array([1, 0, 0, 0])
    >>> assert(np.all(expected == labels))
    >>> update_labels(1, adj_mat, labels)
    >>> expected = np.array([1, 2, 0, 0])
    >>> assert(np.all(expected == labels))
    >>> update_labels(2, adj_mat, labels)
    >>> expected = np.array([1, 2, 1, 0])
    >>> assert(np.all(expected == labels))
    >>> update_labels(3, adj_mat, labels)
    >>> expected = np.array([1, 2, 1, 2])
    >>> assert(np.all(expected == labels))
    """

    roi_neighbors_cond = adj_mat[:, roi_ix] > 0

    if roi_ix == 3:
        print(np.flatnonzero(roi_neighbors_cond))

    if roi_ix == 38:
        print(np.flatnonzero(roi_neighbors_cond))

    roi_neighbors_labels = labels[roi_neighbors_cond]

    roi_label = get_new_label(roi_neighbors_labels)
    labels[roi_ix] = roi_label


def get_new_label(labels):
    """
    >>> import numpy as np
    >>> labels = np.array([0, 1, 2, 3])
    >>> expected = 4
    >>> assert expected == get_new_label(labels)
    >>> labels = np.array([2, 0, 3, 10])
    >>> expected = 1
    >>> assert expected == get_new_label(labels)
    """
    labels = set(labels)

    def smallest_new_label(curr_test=1):
        return curr_test if curr_test not in labels else smallest_new_label(curr_test+1)

    new_label = smallest_new_label()

    assert new_label not in labels
    return new_label


def verify_valid_labels(adj_mat, labels):
    """
    >>> import numpy as np
    >>> adj_mat = np.zeros((4,4))
    >>> neighbors = [(0, 3), (0, 1)]
    >>> for n in neighbors:
    ...  adj_mat[n] = 1
    ...  adj_mat[n[::-1]] = 1
    >>> labels = np.array([1, 2, 3, 4])
    >>> assert verify_valid_labels(adj_mat, labels)[0]
    >>> labels = np.array([1, 2, 3, 1])
    >>> assert not verify_valid_labels(adj_mat, labels)[0]
    """
    num_rois = adj_mat.shape[1]

    def is_valid(roi_ix = 0):
        if roi_ix == num_rois:
            return (True, None)

        neighbors = adj_mat[:, roi_ix] > 0
        label = labels[roi_ix]
        neighbor_labels = labels[neighbors]

        same_neighbors = lambda: np.flatnonzero(neighbors)[neighbor_labels == label]

        if roi_ix == 3:
            print(np.flatnonzero(neighbors))

        return (False, roi_ix, same_neighbors()) if label in neighbor_labels else is_valid(roi_ix+1)

    return is_valid()


def verify_valid_adj_mat(adj_mat):
    """
    >>> import numpy as np
    >>> x = np.array([[0, 1], [1, 0]])
    >>> assert verify_valid_adj_mat(x)[0]
    >>> x[1][0] = 0
    >>> assert not verify_valid_adj_mat(x)[0]
    """
    num_rois = adj_mat.shape[1]

    def is_valid(roi_ix=0):
        if roi_ix == num_rois:
            return (True, None)

        neighbors_cond = adj_mat[:, roi_ix] > 0
        neighbors = np.flatnonzero(neighbors_cond)

        mismatch_cond = adj_mat[roi_ix, neighbors_cond] == 0

        return (False, roi_ix, neighbors[mismatch_cond]) if np.any(mismatch_cond) else is_valid(roi_ix+1)

    return is_valid()



if __name__ == "__main__":
    import doctest
    doctest.testmod()