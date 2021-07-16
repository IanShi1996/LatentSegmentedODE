import numpy as np


def n_true_positive(true_cp, pred_cp, bound):
    """ Get number of true positive detected changepoints.

    Return the number of predicted changepoints which are true positives. A
    margin of error defined in bound is allowed on either side of each
    changepoint.

    Args:
        true_cp (list of int): Indices of ground truth changepoints.
        pred_cp (list of int): Indices of predicted changepoints.
        bound (int): Margin of error when considering true positives.

    Returns:
        int: Count of true positive predicted changepoints.
    """
    correct = 0
    for tcp in true_cp:
        for pcp in pred_cp:
            if np.abs(tcp - pcp) <= bound:
                correct += 1
                break
    return correct 


def f1_score(true_cp, pred_cp, bound=10):
    """Compute F1 score between two segmentations.

    The F1 metric is computed as the harmonic mean between precision and recall
    of true and predicted segmentations. Allows for margin of error when
    calculating true positives.

    Set of indices should include last timepoint as changepoint.

    Args:
        true_cp (list of int): Indices of ground truth changepoints.
        pred_cp (list of int): Indices of predicted changepoints.
        bound (int): Margin of error when considering true positives.

    Returns:
        float: F1 score of changepoint prediction.
    """
    true_pos = n_true_positive(true_cp, pred_cp, bound)
    precision = true_pos / len(pred_cp)
    recall = true_pos / len(true_cp)

    return 2 * ((precision * recall) / (precision + recall))


def hausdorff(true_cp, pred_cp):
    """Compute Hausdorff metric between two segmentations.

    The Hausdorff metric represents the maximal error distance between the two
    segmentations. For further detail, see:
    https://centre-borelli.github.io/ruptures-docs/user-guide/metrics/hausdorff/

    Set of indices should include last timepoint as changepoint.

    Args:
        true_cp (list of int): Indices of ground truth changepoints.
        pred_cp (list of int): Indices of predicted changepoints.

    Returns:
        float: Hausdorff metric of predicted changepoints.
    """
    tcp_dist = []
    for tcp in true_cp:
        min_dist = np.inf
        for pcp in pred_cp:
            if np.abs(tcp - pcp) < min_dist:
                min_dist = np.abs(tcp - pcp)
        tcp_dist.append(min_dist)
    max_tcp_dist = max(tcp_dist)
    
    pcp_dist = []
    for pcp in pred_cp:
        min_dist = np.inf
        for tcp in true_cp:
            if np.abs(tcp - pcp) < min_dist:
                min_dist = np.abs(tcp - pcp)
        pcp_dist.append(min_dist)
    max_pcp_dist = max(pcp_dist)
    
    return max(max_pcp_dist, max_tcp_dist)


def annotation_error(true_cp, pred_cp):
    """Compute annotation error of two segmentations.

    Annotation error is the difference between count of predicted changepoints
    and true changepoints.

    Set of indices should include last timepoint as changepoint.

    Args:
        true_cp (list of int): Indices of ground truth changepoints.
        pred_cp (list of int): Indices of predicted changepoints.

    Returns:
        float: Difference in count between true and predicted changepoints.
    """
    return np.abs(len(true_cp) - len(pred_cp))


def evaluate_metrics(metrics, pred_cp, true_cp):
    """Evaluates set of segmentation on selected metrics

    Args:
        metrics (list of functions): List containing metric functions.
        pred_cp (list of list of int): List of predicted changepoints.
        true_cp (list of list of int): List of ground truth changepoints.

    Returns:
        dict: Map of selected metric to list of computed values.
    """
    metric_data = {metric.__name__: [] for metric in metrics}

    for i in range(len(pred_cp)):
        for metric in metrics:
            out = metric(pred_cp[i], true_cp[i])
            metric_data[metric.__name__].append(out)
    return metric_data
