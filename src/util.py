"""utility function"""
import math
import numpy as np
import scipy.signal
import librosa
from mindspore import ops, nn
from mindspore.nn import TrainOneStepWithLossScaleCell
from mindspore import RowTensor

_grad_scale = ops.composite.MultitypeFuncGraph("grad_scale")
reciprocal = ops.operations.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.functional.cast(reciprocal(scale), ops.functional.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * ops.functional.cast(reciprocal(scale), ops.functional.dtype(grad.values)),
                     grad.dense_shape)


_grad_overflow = ops.composite.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.operations.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


@_grad_overflow.register("RowTensor")
def _tensor_grad_overflow_row_tensor(grad):
    return grad_overflow(grad.values)


class ClipGradients(nn.Cell):
    """
    Clip gradients.
    Args:
        grads (tuple[Tensor]): Gradients.
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.

    Returns:
        tuple[Tensor], clipped gradients.
    """
    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = ops.operations.Cast()
        self.dtype = ops.operations.DType()

    def construct(self, grads, clip_type, clip_value):
        """clip gradients"""
        if clip_type not in(0, 1):
            return grads

        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = ops.composite.clip_by_value(grad, self.cast(ops.functional.tuple_to_array((-clip_value,)), dt),
                                                self.cast(ops.functional.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(ops.functional.tuple_to_array((clip_value,)), dt))
            new_grads = new_grads + (t,)
        return new_grads


class TrainOneStepWithLossScaleCellv2(TrainOneStepWithLossScaleCell):
    """
    Network training with loss scaling.
    """
    def __init__(self, network, optimizer, scale_sense):
        super(TrainOneStepWithLossScaleCellv2, self).__init__(
            network=network, optimizer=optimizer, scale_sense=scale_sense)
        self.clip_gradients = ClipGradients()

    def construct(self, *inputs):
        """loss scale training"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        output = self.network.output

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = ops.composite.ones_like(loss) * \
                             ops.functional.cast(scaling_sens, ops.functional.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.functional.partial(_grad_scale, scaling_sens), grads)
        grads = self.clip_gradients(grads, 0, 1.0)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            loss = ops.functional.depend(loss, self.optimizer(grads))
        return loss, cond, scaling_sens, output


def triangular():
    """
    triangular for cyclic LR. https://arxiv.org/abs/1506.01186
    """
    return 1.0


def triangular2(cycle):
    """
    triangular2 for cyclic LR. https://arxiv.org/abs/1506.01186
    """
    return 1.0 / (2.**(cycle - 1))


def learning_rate_clr_triangle_function(step_size, max_lr, base_lr, iterations):
    """
    get learning rate for cyclic LR. https://arxiv.org/abs/1506.01186
    """
    cycle = math.floor(1 + iterations / (2 * step_size))
    x = abs(iterations / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x)) * triangular()


def audio_to_melspectrogram(audio,
                            sample_rate=16000,
                            n_mels=80,
                            hop_length=160,
                            win_length=400,
                            n_fft=512):
    """
    Compute melspectrogram
    Args:
        audio(numpy.ndarray): audio time-series.
        sample_rate(int): sampling rate of audio.
        n_mels(int): number of Mels to return.
        hop_length(int): number of samples between successive frames.
        n_fft(int): length of the FFT window.
    Returns:
        Mel spectrogram
    """
    spectrogram = librosa.feature.melspectrogram(y=audio,
                                                 sr=sample_rate,
                                                 n_mels=n_mels,
                                                 hop_length=hop_length,
                                                 win_length=win_length,
                                                 n_fft=n_fft,
                                                 window=scipy.signal.windows.hamming,
                                                 fmin=20,
                                                 fmax=7600)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)

    return spectrogram


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
