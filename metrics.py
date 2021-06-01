import numpy as np
from ruptures.metrics import randindex

"""
Define metrics to evaluate segmentation. Implementation uses: https://github.com/deepcharles/ruptures
Key quirk is that changepoint set in ruptures should contain the last timepoint as a changepoint.
"""

def n_true_positive(true_cp, pred_cp, bound):
    correct = 0
    for tcp in true_cp:
        for pcp in pred_cp:
            if np.abs(tcp - pcp) <= bound:
                correct += 1
                break
    return correct 

def f1_score(true_cp, pred_cp, bound=10):
    true_pos = n_true_positive(true_cp, pred_cp, bound)
    precision = true_pos / len(pred_cp)
    recall = true_pos / len(true_cp)

    return 2 * ((precision * recall) / (precision + recall))

def hausdorff(true_cp, pred_cp):
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
    return(np.abs(len(true_cp)-len(pred_cp)))

def evaluate_metrics(metrics, pred_cp, true_cp):
    metric_data = {metric.__name__:[] for metric in metrics}

    for i in range(len(pred_cp)):
        for metric in metrics:
            out = metric(pred_cp[i], true_cp[i])
            metric_data[metric.__name__].append(out)
    return metric_data
