from typing import Any, Dict, Optional, Tuple, Type, Union

from torch import nn
import torch.nn.functional as F
from torch import Tensor
import os

os.environ['HYDRA_FULL_ERROR']='1'

__all__ = ["build_activation"]

# class Mish(nn.Module):
#     def forward(self, input: Tensor) -> Tensor:
#         return nn.Mish(input)

class Mish(nn.Module):
    """Applies the Mish function, element-wise.
    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Mish.png

    Examples::

        >>> m = nn.Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.mish(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


# register activation function here
#   name: module, kwargs with default values
REGISTERED_ACT_DICT: Dict[str, Tuple[Type, Dict[str, Any]]] = {
    "relu": (nn.ReLU, {"inplace": True}),
    "relu6": (nn.ReLU6, {"inplace": True}),
    "leaky_relu": (nn.LeakyReLU, {"inplace": True, "negative_slope": 0.1}),
    "h_swish": (nn.Hardswish, {"inplace": True}),
    "h_sigmoid": (nn.Hardsigmoid, {"inplace": True}),
    "swish": (nn.SiLU, {"inplace": True}),
    "silu": (nn.SiLU, {"inplace": True}),
    "tanh": (nn.Tanh, {}),
    "sigmoid": (nn.Sigmoid, {}),
    "gelu": (nn.GELU, {}),
    "mish": (Mish, {"inplace": True}),
}


def build_activation(act_func_name: Union[str, nn.Module], **kwargs) -> Optional[nn.Module]:
    if isinstance(act_func_name, nn.Module):
        return act_func_name
    if act_func_name in REGISTERED_ACT_DICT:
        act_module, default_args = REGISTERED_ACT_DICT[act_func_name]
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return act_module(**default_args)
    elif act_func_name is None or act_func_name.lower() == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func_name)
