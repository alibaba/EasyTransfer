""" Operations """
import torch
import torch.nn as nn
import genotypes as gt
import math
from torch.nn import ReLU

hid_size = 128


def gelu(x):
  return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


# to make the output length as the same as the input length, 2*p = d(k-1), p is pad, k is window_size, d is dilation.
#

OPS = {
    'none': lambda C_in, C_out, stride, affine: Zero(stride),
    'avg_pool1d_3': lambda C_in, C_out, stride, affine: PoolBN('avg', 3, C_out, stride, padding=1, affine=affine),
    'max_pool1d_3': lambda C_in, C_out, stride, affine: PoolBN('max', 3, C_out, stride, padding=1, affine=affine),
    'skip_connect': lambda C_in, C_out, stride, affine: Identity() if stride == 1 else FactorizedReduce(C_in, C_out, affine=affine),
    'dil_conv_3': lambda C_in, C_out, stride, affine: TextConv(C_in, C_out, 3, stride, dilation=2, affine=affine),
    'dil_conv_5': lambda C_in, C_out, stride, affine: TextConv(C_in, C_out, 5, stride, dilation=2, affine=affine),
    'dil_conv_7': lambda C_in, C_out, stride, affine: TextConv(C_in, C_out, 7, stride, dilation=2, affine=affine),
    'std_conv_3': lambda C_in, C_out, stride, affine: TextConv(C_in, C_out, 3, stride, dilation=1, affine=affine),
    'std_conv_5': lambda C_in, C_out, stride, affine: TextConv(C_in, C_out, 5, stride, dilation=1, affine=affine),
    'std_conv_7': lambda C_in, C_out, stride, affine: TextConv(C_in, C_out, 7, stride, dilation=1, affine=affine),
}


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, kernel_size, C, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool1d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool1d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm1d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    Conv - ReLU - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding),
            # nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            ReLU(),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class TextConv(nn.Module):
    """ KIM CNN
    ReLU - Conv - BN
    """
    # kernel_size n indicates n-gram window size
    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, affine=True):
        super().__init__()
        # pad to make the input length as the same as the output length
        pad = int((kernel_size - 1) * dilation / 2)
        assert 2 * pad == dilation * (kernel_size - 1)  # (dilation is odd and kernel_size is even) is forbidden
        self.net = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, dilation=dilation, padding=pad),
            ReLU(),
            # nn.MaxPool1d(kernel_size, stride=1, padding=pad),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        # x = gelu(x)
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
