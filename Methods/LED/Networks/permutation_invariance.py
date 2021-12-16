#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module


class PermutationInvarianceModule(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::
        >>> m = PermutationInvarianceModule(20)
        >>> input = torch.randn(32, 1000)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([32, 1000, 20])
        torch.Size([32, 20])
    """
    __constants__ = ['bias', 'features']

    def __init__(self, channels, features, layers, bias=True):
        super(PermutationInvarianceModule, self).__init__()
        self.channels = channels
        self.features = features

        self.weight = Parameter(
            torch.DoubleTensor(self.features, self.channels))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(self.channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        input = input.unsqueeze(2)
        # print(input.size())
        output = F.linear(input, self.weight, self.bias)
        # output = torch.mean(output, 1)
        # output = torch.nn.functional.selu(output)
        return output

    def extra_repr(self):
        return 'features={}, channels={}, bias={}'.format(
            self.features, self.channels, self.bias is not None)


if __name__ == "__main__":

    m = PermutationInvarianceModule(20)
    input = torch.randn(32, 1000)
    output = m(input)
    print(output.size())
