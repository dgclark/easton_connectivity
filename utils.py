from __future__ import division

import re

import numpy as np
import pandas as pd


def calc_residuals(independent, dependent):
    num_ind_subjects, num_ind_features = independent.shape
    num_dep_subjects, num_dep_features = dependent.shape
    assert num_ind_subjects == num_dep_subjects
    num_subjects = num_ind_subjects

    sln, _, _, _ = np.linalg.lstsq(independent, dependent)
    assert sln.shape == (num_ind_features, num_dep_features)

    residuals = dependent - np.dot(independent, sln)
    assert residuals.shape == (num_subjects, num_dep_features)

    return residuals


def is_roi_col(col_name):
    """
    >>> assert is_roi_col("L123")
    >>> assert not is_roi_col("L23a")
    >>> assert not is_roi_col("a123")
    >>> assert is_roi_col("R5")
    """
    return re.match(r"[L|R][0-9]+$", col_name)


def correct_rois_for_nuisance(output_f=None):
    meta_data = pd.read_csv("data/animal_scores.csv")
    meta_data = meta_data.set_index(meta_data.id)
    roi_data = pd.read_csv("data/ROI_matrix.txt", sep='\t')
    roi_data = roi_data.set_index(roi_data.id)

    num_subjects = min(roi_data.shape[0], meta_data.shape[0])
    num_cols = roi_data.shape[1] + meta_data.shape[1]

    all_data = pd.concat((meta_data, roi_data), join='inner', axis=1)
    assert all_data.shape == (num_subjects, num_cols)

    def verify_index(df):
        assert np.all(df.index == all_data.index)

    rois_only = all_data[[c for c in roi_data.columns if is_roi_col(c)]]
    assert rois_only.shape[0] == num_subjects
    verify_index(rois_only)

    cols = ['edu', 'age', 'sex']
    nuisance = all_data[cols]
    nuisance['age_x_sex'] = nuisance.age * nuisance.sex
    nuisance['mean_roi'] = np.average(rois_only, axis=1)
    assert nuisance.shape == (num_subjects, 5)
    verify_index(nuisance)

    normalized_rois = calc_residuals(nuisance, rois_only)
    assert normalized_rois.shape == rois_only.shape

    for c in normalized_rois.columns:
        normalized_rois[c + '_normalized'] = normalized_rois[c]
        del normalized_rois[c]

    normalized_rois = pd.concat((normalized_rois, rois_only, nuisance),
                                join='inner', axis=1)
    assert normalized_rois.shape[0] == num_subjects
    verify_index(normalized_rois)


    if output_f is not None:
        normalized_rois.to_csv(output_f)

    return normalized_rois


def load_rois():
    return pd.read_csv("data/ROI_matrix.txt", sep="\t")


def calc_roi_corrs(rois=None):
    rois = load_rois().iloc[:, 3:] if rois is None else rois
    roi_cols = rois.columns

    cov = np.corrcoef(rois.T)

    num_rois = len(roi_cols)
    assert cov.shape == (num_rois, num_rois)

    return pd.DataFrame(cov,
                        index=roi_cols,
                        columns=roi_cols)

def apply_both(fn, lows, highs):
    return fn(lows), fn(highs)