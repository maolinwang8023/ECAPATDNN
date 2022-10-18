"""ECAPATDNN model"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer
mindspore.set_seed(0)


class BatchNorm1d(nn.Cell):
    def __init__(self, input_size):
        super(BatchNorm1d, self).__init__()
        self.batchnorm = nn.BatchNorm2d(input_size)
        self.squeeze = ops.Squeeze(-1)
        self.expandDims = ops.ExpandDims()

    def construct(self, x):
        return self.squeeze(self.batchnorm(self.expandDims(x, -1)))


class TDNNBlock(nn.Cell):
    """
    An implementation of TDNN layer.
    Args:
        in_channels(int): The number of input channels.
        out_channels(int): The number of output channels.
        kernel_size(int): The kernel size of the TDNN layer.
        dilation(int): The dilation of the Res2Net layer.
        activation: A class for constructing the activation layer.
    Returns:
        the output of TDNN layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=nn.ReLU):
        super(TDNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              has_bias=True,
                              weight_init='he_uniform',
                              bias_init='truncatedNormal')
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def construct(self, x):
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(nn.Cell):
    """
    An implementation of the Res2Net layer.
    Args:
        in_channels(int): The number of input channels.
        out_channels(int): The number of output channels.
        scale(int): The scale of the Res2Net layer.
        kernel_size(int): The kernel size of the Res2Net layer.
        dilation(int): The dilation of the Res2Net layer.
    Returns:
        the output of Res2Net layer.
    """
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.CellList([
            TDNNBlock(
                in_channel,
                hidden_channel,
                kernel_size=kernel_size,
                dilation=dilation,
                )
            for i in range(scale - 1)
            ])
        self.scale = scale

        self.cat = ops.Concat(axis=1)
        self.split = ops.Split(1, scale)

    def construct(self, x):
        """the Res2Net layer"""
        y = []
        spx = self.split(x)
        y_i = x
        for i, x_i in enumerate(spx):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = self.cat(y)
        return y


class SEBlock(nn.Cell):
    """
    An implementation of squeeze and excitation layer.
    Args:
        in_channels(int): The number of input channels.
        se_channels(int): The number of output channels after squeeze.
        out_channels(int): The number of output channels.
    Returns:
        the output of squeeze and excitation layer.
    """
    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=se_channels,
                               kernel_size=1,
                               has_bias=True,
                               weight_init='he_uniform',
                               bias_init='truncatedNormal')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=se_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               has_bias=True,
                               weight_init='he_uniform',
                               bias_init='truncatedNormal')
        self.sigmoid = nn.Sigmoid()

        self.expandDim = ops.ExpandDims()

    def construct(self, x, lengths=None):
        """the squeeze and excitation layer"""
        if lengths is not None:
            length = lengths * x.shape[-1]
            max_len = length.max()
            expand = ops.BroadcastTo((len(length), max_len))
            mask = expand(nn.Range(0, max_len, 1)) < self.expandDim(length, 1)
            mask = self.expandDim(mask, 1)
            total = mask.sum(axis=2, keepdims=True)
            s = (x * mask).sum(axis=2, keepdims=True) / total
        else:
            s = x.mean((2), True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class MaskedFill(nn.Cell):
    """
    Creates a binary mask for each sequence
    Args:
        value(int): the length of each sequence.
    Returns:
        the output of masking input.
    """
    def __init__(self, value):
        super(MaskedFill, self).__init__()
        self.value = Tensor([value], mindspore.float32)
        self.minusend = Tensor([1.0], mindspore.float32)
        self.sub = ops.Sub()
        self.mul = ops.Mul()

    def construct(self, inputs: Tensor, mask: Tensor):
        masked = self.sub(self.minusend, mask)
        adder = self.mul(mask, self.value)
        inputs = self.mul(masked, inputs)
        output = inputs + adder
        return output


class AttentiveStatisticsPooling(nn.Cell):
    """
    An implementation of the attentive statistic pooling layer for each channel
    Args:
        channels(int): The number of input channels.
        attention_channels(int): The number of attention channels.
        global_context(bool): Whether to look at global properties of the utterance.
    Returns:
        the output of ASP layer.
    """
    def __init__(self, channels, attention_channels=128, global_context=False):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(in_channels=attention_channels,
                              out_channels=channels,
                              kernel_size=1,
                              has_bias=True,
                              weight_init='he_uniform',
                              bias_init='truncatedNormal')

        self.sqrt = ops.Sqrt()
        self.pow = ops.Pow()
        self.expandDim = ops.ExpandDims()
        self.softmax = ops.Softmax(axis=2)
        self.cat = ops.Concat(axis=1)
        self.ones = ops.Ones()
        self.tile = ops.Tile()
        self.masked_fill = MaskedFill(float("-inf"))

    def construct(self, x, lengths=None):
        """attentive statistic pooling layer"""
        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = self.sqrt(((m * self.pow((x -self.expandDim(mean, dim)), 2)).sum(dim)).clip(eps, None))
            return mean, std

        attn = 0
        mask = None
        if self.global_context:
            if lengths is None:
                lengths = self.ones((x.shape[0],), mindspore.float32)
                length = lengths * x.shape[-1]
                max_len = length.max()
                expand = ops.BroadcastTo((len(length), max_len))
                mask = expand(nn.Range(0, max_len, 1)) < self.expandDim(length, 1)
                mask = self.expandDim(mask, 1)
                total = mask.sum(axis=2, keepdims=True)
                mean, std = _compute_statistics(x, mask / total)
                mean = mindspore.numpy.tile(self.expandDim(mean, 2), (1, 1, x.shape[-1]))
                std = mindspore.numpy.tile(self.expandDim(mean, 2), (1, 1, x.shape[-1]))
                attn = self.cat(x, mean, std)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        if mask is not None:
            attn = self.masked_fill(attn, mask == 0)

        attn = self.softmax(attn)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = self.cat((mean, std))
        pooled_stats = self.expandDim(pooled_stats, 2)

        return pooled_stats


class SERes2NetBlock(nn.Cell):
    """
    An implementation of TDNN-Res2Net-TDNN-SEBlock
    Args:
        in_channels(int): The number of input channels.
        out_channels(int): The number of output channels.
        res2net_scale(int): The scale of the Res2Net layer.
        kernel_size(int): The kernel size of the TDNN layer.
        dilation(int): The dilation of the Res2Net layer.
        activation: A class for constructing the activation layer.
    Returns:
        the output of TDNN-Res2Net-TDNN-SEBlock layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 res2net_scale=8,
                 se_channels=128,
                 kernel_size=1,
                 dilation=1,
                 activation=nn.ReLU):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(in_channels,
                               out_channels,
                               kernel_size=1,
                               dilation=1,
                               activation=activation)
        self.res2net_block = Res2NetBlock(out_channels,
                                          out_channels,
                                          res2net_scale,
                                          kernel_size,
                                          dilation)
        self.tdnn2 = TDNNBlock(out_channels,
                               out_channels,
                               kernel_size=1,
                               dilation=1,
                               activation=activation)
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      has_bias=True,
                                      weight_init='he_uniform',
                                      bias_init='truncatedNormal')

    def construct(self, x, lengths=None):
        """TDNN-Res2Net-TDNN-SEBlock Layer"""
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual


class ECAPA_TDNN(nn.Cell):
    """
    The implementation of the speaker embedding model
    Args:
        input_size(int): Number of channels.
        lin_neurons(int): Number of neurons in linear layers.
        activation: A class for constructing the activation layers.
        kernel_sizes(tuple): Tuple of kernel sizes for each layer.
        dilations(tuple): Tuple of dilations for kernels in each layer.
        attention_channels(int): The channels of the ASP layer.
        res2net_scale(int): The scale of the Res2Net layer.
        se_channels(int): The channels of the SE layer.
        global_context(bool): Whether to look at global properties of the utterance.
    Returns:
        embedding
    """
    def __init__(self,
                 input_size,
                 lin_neurons=192,
                 activation=nn.ReLU,
                 channels=(512, 512, 512, 512, 1536),
                 kernel_sizes=(5, 3, 3, 3, 1),
                 dilations=(1, 2, 3, 4, 1),
                 attention_channels=128,
                 res2net_scale=8,
                 se_channels=128,
                 global_context=False):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.CellList()

        # TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(channels[-1],
                             channels[-1],
                             kernel_sizes[-1],
                             dilations[-1],
                             activation)

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(channels[-1],
                                              attention_channels=attention_channels,
                                              global_context=global_context)
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = nn.Conv1d(in_channels=channels[-1] * 2,
                            out_channels=lin_neurons,
                            kernel_size=1,
                            has_bias=True,
                            weight_init='he_uniform',
                            bias_init='truncatedNormal')

        self.cat = ops.Concat(axis=1)

    def construct(self, x, lengths=None):
        """Compute embedding"""
        xl = []
        for layer in self.blocks:
            x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        layer_tmp = []
        for idx in range(1, len(xl)):
            layer_tmp.append(xl[idx])
        x = self.cat(layer_tmp)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        return x.squeeze()


class Classifier(nn.Cell):
    """
    Compute the cosine similarity on the top of features
    Args:
        input_size(int): Number of channels.
        lin_blocks(int):  Number of linear layers.
        lin_neurons(int): Number of neurons in linear layers.
        out_neurons(int): Number of classes.
    Returns:
        the cosine similarity
    """
    def __init__(self, input_size, lin_blocks=0, lin_neurons=192, out_neurons=5994):
        super().__init__()
        self.blocks = nn.CellList()

        for _ in range(lin_blocks):
            self.blocks.extend(
                [
                    BatchNorm1d(input_size=input_size),
                    nn.Dense(in_channels=input_size, out_channels=lin_neurons),
                ]
            )
            input_size = lin_neurons
        input_size = lin_neurons
        # Final Layer
        self.weight = Parameter(initializer("xavier_uniform", shape=(out_neurons, input_size),
                                            dtype=mindspore.float32).init_data())
        self.norm = ops.L2Normalize(axis=1)
        self.matmul = ops.MatMul()
        self.expand_dims = ops.ExpandDims()

    def construct(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.matmul(self.norm(x), self.norm(self.weight).transpose())
        return x


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)
    input_feats = Tensor(np.ones([100, 80, 202]), mindspore.float32)
    compute_embedding = ECAPA_TDNN(80, channels=(1024, 1024, 1024, 1024, 1024 * 3), lin_neurons=192)
    outputs = compute_embedding(input_feats)
    print(outputs.shape)
