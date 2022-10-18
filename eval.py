"""eval"""
import argparse
import os
import tqdm
import soundfile
import mindspore
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore import load_checkpoint, load_param_into_net
import numpy
from scipy.spatial.distance import cosine

from src.model import ECAPA_TDNN
from src.metrics import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from src.util import audio_to_melspectrogram, emb_mean
from src.config import Config_gpu, Config_ascend
mindspore.set_seed(0)


def eval_network(eval_list, eval_path, model):
    """
    Eval network
    Args:
        eval_list(str): The path of the evaluation list.
        eval_path(str): The path of the evaluation data.
        model: The ECAPATDNN network.
    Returns:
        EER and minDCF
    """
    model.set_train(False)
    normalize = ops.L2Normalize(axis=1)
    reshape = ops.Reshape()
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()
    for line in lines:
        files.append(line.split()[1])
        files.append(line.split()[2])
    setfiles = list(set(files))
    setfiles.sort()

    for _, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
        audio, _ = soundfile.read(os.path.join(eval_path, file))
        length = 1200 * 160 - 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(numpy.random.random() * (audio.shape[0] - length))
        audio = audio[start_frame : start_frame + length]
        data = numpy.stack([audio], axis=0)
        data = audio_to_melspectrogram(data)
        data = data - numpy.mean(data, axis=-1, keepdims=True)
        data = Tensor(data, mindspore.float32)
        embedding = model(data)
        embedding = reshape(embedding, (1, embedding.shape[0]))
        embedding = normalize(embedding)
        embeddings[file] = embedding

    glob_mean = Tensor([0])
    cnt = 0
    enroll_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, embeddings)
    enroll_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, embeddings)
    enroll_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, embeddings)
    embeddings = enroll_dict_mean
    scores, labels = [], []

    for line in lines:
        embedding_11 = embeddings[line.split()[1]]
        embedding_21 = embeddings[line.split()[2]]
        score = 1 - cosine(embedding_11.copy().asnumpy(), embedding_21.copy().asnumpy())
        scores.append(score)
        labels.append(int(line.split()[0]))

    # Coumpute EER and minDCF
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    return EER, minDCF


def get_args():
    """ get args"""
    parser = argparse.ArgumentParser(description='Train CTRL')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend.')
    parser.add_argument('--model_path', type=str, default=None, help='the path of eval check_point.')
    parser.add_argument('--device_target', type=str, default="Ascend", help='device target.')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.device_target == "GPU":
        cfg = Config_gpu()
    else:
        cfg = Config_ascend()
    cfg.device_id = args.device_id
    if args.model_path is not None:
        cfg.model_path = args.model_path

    context.set_context(mode=cfg.mode, device_target=cfg.device_target, device_id=cfg.device_id)
    network = ECAPA_TDNN(input_size=cfg.in_channels,
                         channels=(cfg.channels, cfg.channels, cfg.channels, cfg.channels, cfg.channels * 3),
                         lin_neurons=cfg.emb_size)

    param_dict = load_checkpoint(cfg.model_path)
    load_param_into_net(network, param_dict)
    eer, mindcf = eval_network(cfg.eval_list, cfg.eval_path, network)
    print("EER = %2.2f%%\n" % eer)
