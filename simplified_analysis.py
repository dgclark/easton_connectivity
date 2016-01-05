from __future__ import division

import numpy as np
import pandas as pd

import strong_rois


def main_part(f_name, field, num_perms, cut_off=.95, verbose=True, permutations_dict=None):

    subjects_dict = {}
    subjects_dict['low'], subjects_dict['high'] = split_ids(f_name, field)
    rois = pd.read_csv("ROI_matrix.txt", sep="\t")
    roi_cols = rois.columns[3:]

    ret = {}

    def get_subject_rois(subject_ids):
        id_rows = rois.id.isin(subject_ids)
        return rois.loc[id_rows, roi_cols].get_values()

    def calc_roi_corrs(subject_rois):
        cov = np.corrcoef(subject_rois.T)

        num_rois = len(roi_cols)
        assert cov.shape == (num_rois, num_rois)

        return pd.DataFrame(cov,
                            index=roi_cols,
                            columns=roi_cols)

    def calc_permutations(subject_rois, subject_type):
        if permutations_dict is None:
            sorted_permutations = strong_rois.sorted_permutations(subject_rois, num_perms, verbose)
        else:
            sorted_permutations = permutations_dict[subject_type]

        return sorted_permutations

    for subject_type, subject_ids in subjects_dict.items():
        subject_rois = get_subject_rois(subject_ids)

        sorted_permutations = calc_permutations(subject_rois, subject_type)

        roi_corrs = calc_roi_corrs(subject_rois)

        ret[subject_type] = {}
        ret[subject_type]['valid_connections'] = \
            strong_rois.connections_above(sorted_permutations, cut_off, roi_corrs)
        ret[subject_type]['orig'] = roi_corrs
        ret[subject_type]['sorted_permutations']=sorted_permutations

    return ret


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