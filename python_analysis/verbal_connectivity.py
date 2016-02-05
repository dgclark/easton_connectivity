from __future__ import division

import networkx as nx
import numpy as np
import pandas as pd

import median_split
import utils


def calc_graph_metrics(graph_count=1000, verbose=0):

    animal_scores = pd.read_csv('data/animal_scores.csv')
    num_subjects, num_cols = animal_scores.shape
    field = 'raw'

    def create_return_dict():
        ret = dict(low=[], high=[], median=[])

        def update(f, val):
            ret[f].append(val)

        return ret, update

    ids, update_ids = create_return_dict()
    metrics, update_metrics = create_return_dict()

    last_print = -1

    rois = prepped_rois()

    def ids_to_metrics(ids):
        graph = ids_to_graph(ids, rois)

        metrics = dict(local=calc_local_graph_metrics(graph))
        metrics['global'] = calc_global_graph_metrics(graph, metrics['local'])
        return metrics

    for i in range(graph_count):
        if (i - last_print) == verbose:
            last_print = i
            print "number %s out of %s" % (i + 1, graph_count)

        sample_rows = np.random.choice(num_subjects, num_subjects)
        animal_score_samples = animal_scores.loc[sample_rows, :]
        assert animal_score_samples.shape == (num_subjects, num_cols)

        low_ids, high_ids, median = median_split.split_ids(animal_score_samples, field)

        low_metrics, high_metrics = utils.apply_both(ids_to_metrics, low_ids, high_ids)

        update_ids('low', low_ids)
        update_ids('high', high_ids)
        update_ids('median', median)

        update_metrics('low', low_metrics)
        update_metrics('high', high_metrics)
        update_metrics('median', median)

    return dict(ids=ids, metrics=metrics)


def ids_to_graph(ids, rois=None):
    if rois is None:
        rois = prepped_rois()

    subj_rois = rois.loc[rois.index.isin(ids), :]
    corrs = utils.calc_roi_corrs(subj_rois)
    return corr_to_graph(corrs)


def prepped_rois():
    n = '_normalized'
    rois = utils.correct_rois_for_nuisance().filter(like=n)
    rois.rename(columns={c: c[:-len(n)] for c in rois.columns}, inplace=True)
    return rois


def corr_to_graph(roi_corrs, copy_corrs=False):
    """
    >>> import pandas as pd
    >>> import numpy as np
    >>> corrs = pd.DataFrame(np.random.rand(2,2))
    >>> corrs.index = ['A', 'B']
    >>> corrs.columns = ['A', 'B']
    >>> graph = corr_to_graph(corrs)
    >>> ab = graph['A']['B']
    >>> wt, prox, dist = ab['weight'], ab['proximity'], ab['distance']
    >>> assert wt == corrs['B']['A'] #upper triangular
    >>> assert prox == wt
    >>> assert dist == 1 - wt
    >>> assert len(graph) == 2
    """
    roi_corrs = create_convertible_corr_df(roi_corrs, copy_corrs)
    return nx.from_pandas_dataframe(roi_corrs, 'source', 'target',
                                    edge_attr=['distance', 'proximity', 'weight'])


def corr_to_dist_prox(corr_series):
    """
    >>> import pandas as pd
    >>> import numpy as np
    >>> corr = pd.Series([.2, -.8])
    >>> (dist, prox) = corr_to_dist_prox(corr)
    >>> expected_dist = np.array([.8, 1])
    >>> expected_prox = np.array([.2, 0])
    >>> assert np.all(dist == expected_dist)
    >>> assert np.all(prox == expected_prox)
    """
    prox = corr_series.copy()
    prox[prox < 0] = 0
    dist = 1 - prox
    return dist, prox


def create_convertible_corr_df(roi_corrs, copy_corrs=False):
    if copy_corrs:
        roi_corrs = roi_corrs.copy()

    roi_corrs['source'] = roi_corrs.index
    num_rois = roi_corrs.shape[0]

    roi_corrs = pd.melt(roi_corrs, id_vars='source')
    roi_corrs['distance'], roi_corrs['proximity'] = corr_to_dist_prox(roi_corrs.value)

    roi_corrs.rename(columns={'variable': 'target', 'value': 'weight'}, inplace=True)
    assert roi_corrs.shape == (num_rois*num_rois, 5)
    assert np.all(
            [c in roi_corrs.columns for c in ['source', 'target', 'distance', 'proximity', 'weight']])

    return roi_corrs


def calc_global_graph_metrics(graph, local_metrics):
    ret = dict()

    avg_path_length = nx.average_shortest_path_length(graph, weight='distance')
    avg_clustering = np.mean(local_metrics['clustering'].values())
    ret['small_worldness'] = avg_clustering/avg_path_length

    return ret


@utils.memoize
def node_order(max_roi):
    return {rl+str(roi): (rl == 'R')*(max_roi+1) + roi
            for rl in 'RL' for roi in range(max_roi + 1)}


def calc_local_graph_metrics(graph):
    ret = dict()
    ret['degree'] = graph.degree(weight='weight')
    ret['bw_centrality'] = nx.betweenness_centrality(graph, weight='distance')
    ret['clustering'] = nx.clustering(graph, weight='proximity')
    return ret


if __name__ == "__main__":
    import doctest
    doctest.testmod()
