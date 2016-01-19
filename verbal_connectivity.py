from __future__ import division

import networkx as nx
import numpy as np
import pandas as pd

import median_split
import utils

def calc_graphs(graph_count=1000, verbose=0, include_corrs=False):
    rois = utils.correct_rois_for_nuisance().filter(like='normalized')

    animal_scores = pd.read_csv('data/animal_scores.csv')
    num_subjects, num_cols = animal_scores.shape
    field = 'raw'

    def create_return_dict():
        ret = dict(low=[], high=[])
        def update(field, val):
            ret[field].append(val)
        return ret, update

    graphs, update_graphs = create_return_dict()
    corrs, update_corrs = create_return_dict()

    last_print = -1
    for i in range(graph_count):
        if (i - last_print) == verbose:
            last_print = i
            print "number %s out of %s" % (i + 1, graph_count)
        sample_rows = np.random.choice(num_subjects, num_subjects)
        animal_score_samples = animal_scores.loc[sample_rows, :]
        assert animal_score_samples.shape == (num_subjects, num_cols)

        get_subj_rois = lambda ids: \
            rois.loc[rois.index.isin(ids), :]

        low_ids, high_ids = median_split.split_ids(animal_score_samples, field)

        low_rois, high_rois = utils.apply_both(get_subj_rois, low_ids, high_ids)

        low_corrs, high_corrs = utils.apply_both(utils.calc_roi_corrs,
                                                 low_rois, high_rois)
        if include_corrs:
            update_corrs('low', low_corrs)
            update_corrs('high', high_corrs)

        low_graph, high_graph = utils.apply_both(corr_to_graph,
                                                 low_corrs, high_corrs)

        update_graphs('low', low_graph)
        update_graphs('high', high_graph)

    return dict(graphs=graphs, corrs=corrs) if include_corrs else graphs


def corr_to_graph(roi_corrs):
    """
    >>> import pandas as pd
    >>> import numpy as np
    >>> corrs = pd.DataFrame(np.random.rand(2,2))
    >>> corrs.index = ['A', 'B']
    >>> corrs.columns = ['A', 'B']
    >>> graph = corr_to_graph(corrs)
    >>> assert graph['A']['B'] == {'weight': corrs['B']['A']} #upper triangular
    >>> assert len(graph) == 2
    """
    g = nx.Graph()

    is_lower = lambda r, c: c > r
    row_cols = [(r, c)
                for rx, r in enumerate(roi_corrs.index)
                for cx, c in enumerate(roi_corrs.columns)
                if is_lower(rx, cx)]

    for (r, c) in row_cols:
            val = roi_corrs[c][r]
            g.add_edge(r, c, {'weight': val})

    return g



def calc_graph_metrics(graph):
    raise NotImplementedError


if __name__ == "__main__":
    import doctest
    doctest.testmod()