"""310 eval"""
import os
import sys
from operator import itemgetter
import numpy
from scipy.spatial.distance import cosine
from mindspore import Tensor
from sklearn import metrics


def tuneThresholdfromScore(embedding_scores, embedding_labels, target_fa, target_fr=None):
    """Compute equal error rate"""
    fpr, tpr, thresholds_res = metrics.roc_curve(embedding_labels, embedding_scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds_res[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds_res[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return tunedThreshold, eer, fpr, fnr


def ComputeErrorRates(res_scores, res_labels):
    """Compute false-negative rates, false-positive rates and  decision thresholds"""
    sorted_indexes, res_thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(res_scores)],
        key=itemgetter(1)))

    res_labels = [res_labels[i] for i in sorted_indexes]
    res_fnrs = []
    res_fprs = []

    for i in range(0, len(res_labels)):
        if i == 0:
            res_fnrs.append(res_labels[i])
            res_fprs.append(1 - res_labels[i])
        else:
            res_fnrs.append(res_fnrs[i - 1] + res_labels[i])
            res_fprs.append(res_fprs[i - 1] + 1 - res_labels[i])
    fnrs_norm = sum(res_labels)
    fprs_norm = len(res_labels) - fnrs_norm

    res_fnrs = [x / float(fnrs_norm) for x in res_fnrs]

    res_fprs = [1 - x / float(fprs_norm) for x in res_fprs]
    return res_fnrs, res_fprs, res_thresholds


def ComputeMinDcf(res_fnrs, res_fprs, res_thresholds, p_target, c_miss, c_fa):
    """Computes the minimum of the detection cost function."""
    min_c_det = float("inf")
    min_c_det_threshold = res_thresholds[0]
    for i in range(0, len(res_fnrs)):
        c_det = c_miss * res_fnrs[i] * p_target + c_fa * res_fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = res_thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def emb_mean(g_mean, increment, embedding_dict):
    """embedding mean"""
    embedding_dict_mean = dict()
    for utt in embedding_dict:
        if increment == 0:
            g_mean = embedding_dict[utt]
        else:
            weight = 1 / (increment + 1)
            g_mean = (1 - weight) * g_mean + weight * embedding_dict[utt]
        embedding_dict_mean[utt] = embedding_dict[utt] - g_mean
        increment += 1
    return embedding_dict_mean, g_mean, increment


if __name__ == "__main__":
    data_path = sys.argv[1]
    eval_list = os.path.join(data_path, "eval.txt")
    data = "output/"
    embeddings = {}
    lines = open(eval_list).read().splitlines()

    with open(os.path.join(data, 'emb.txt'), 'r') as fp:
        for line in fp:
            emb_file = data + line.strip()
            arr = numpy.fromfile(emb_file, dtype=numpy.float32)
            embeddings[line[:-5]] = arr

    glob_mean = Tensor([0])
    cnt = 0
    emb_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, embeddings)
    emb_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, embeddings)
    emb_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, embeddings)
    embeddings = emb_dict_mean

    scores, labels = [], []
    for line in lines:
        labels.append(int(line.split()[0]))
        spk = line.split()[1]
        spk = spk[0:spk.find('.')]
        spk = spk.replace('/', '_')
        embedding_spk = embeddings[spk]

        test = line.split()[2]
        test = test[0:test.find('.')]
        test = test.replace('/', '_')
        embedding_test = embeddings[test]

        score = 1 - cosine(embedding_spk, embedding_test)
        scores.append(score)

    # Coumpute EER and minDCF
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    print("EER: %2.2f%%" % EER)
