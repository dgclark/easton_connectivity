from __future__ import division

import numpy as np
import pandas as pd

import strong_rois


def main_part(f_name, field, num_perms, verbose=True):

    low_score_ids, high_score_ids = split_ids(f_name, field)

    sorted_perms = lambda ids: sorted_perms_in_ids(ids, num_perms, verbose)

    return {'low':sorted_perms(low_score_ids), 'high':sorted_perms(high_score_ids)}


def verify_length_does_not_impact_thresh(f_name, field, num_samples, num_perms, cut_off, verbose=True):
    low_score_ids, high_score_ids = split_ids(f_name, field)

    num_low_ids = len(low_score_ids)
    num_high_ids = len(high_score_ids)

    more_ids = low_score_ids if num_low_ids > num_high_ids else high_score_ids

    more_count = len(more_ids)
    less_count = num_low_ids + num_high_ids - more_count

    calc_cutoff = lambda ids: strong_rois.cutoff_value(
        sorted_perms_in_ids(ids, num_perms, verbose), cut_off)

    i = 0
    cutoffs = np.zeros(num_samples)
    for i in range(num_samples):
        if verbose:
            print "%s sample out of %s" % (i, num_samples)
        ids = np.random.choice(more_ids, less_count, replace=False)
        cutoffs[i] = calc_cutoff(ids)

    return calc_cutoff(more_ids), cutoffs



def split_ids(f_name, field):
    data = pd.read_csv(f_name)
    values = data[field]

    below, above = split_most_even(values)

    low_score_ids = data[below].id
    high_score_ids = data[above].id

    return low_score_ids, high_score_ids


def split_most_even(values):
    """
    >>> import numpy as np
    >>> T = True; F = False
    >>> values = np.array([1, 2, 2, 3, 3, 4])
    >>> expected = np.array([F, F, F, T, T, T])
    >>> actual = split_most_even(values)
    >>> assert(np.all(expected==actual))
    >>> values = np.array([1, 2, 2, 3, 3, 3, 4])
    >>> expected = np.array([F, F, F, T, T, T, T])
    >>> actual = split_most_even(values)
    >>> assert(np.all(expected==actual))
    >>> values = np.array([2, 3, 3, 3, 3, 3, 4])
    >>> expected = np.array([F, T, T, T, T, T, T])
    >>> actual = split_most_even(values)
    >>> assert(np.all(expected==actual))
    >>> values = np.array([2, 3, 3, 3, 3, 3, 4, 4])
    >>> expected = np.array([F, F, F, F, F, T, T])
    >>> actual = split_most_even(values)
    >>> assert(np.all(expected==actual))
    """
    median = np.median(values)
    below = values < median
    above = values > median
    meds = values == median

    num_below = np.sum(below)
    num_above = np.sum(above)
    if (num_below < num_above):
        below = np.logical_or(below, meds)
    else:
        above = np.logical_or(above, meds)

    return (below, above)


def sorted_perms_in_ids(ids, num_perms, verbose):
    rois = pd.read_csv("ROI_matrix.txt", sep="\t")
    rois_with_valid_id = rois.loc[rois.id.isin(ids), rois.columns[3:]].get_values()

    return strong_rois.sorted_permutations(rois_with_valid_id, num_perms, verbose)