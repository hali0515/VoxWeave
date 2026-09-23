# Copyright SpeechBrain contributors.
# Licensed under the Apache License, Version 2.0 (see LICENSE in this directory).
#
# Vendored from https://github.com/speechbrain/speechbrain tag v1.0.3
# (commit 31c1e329048c0380dc7f2acbe680c44a036b6286), file(s):
# speechbrain/dataio/dataio.py, speechbrain/nnet/CNN.py,
# speechbrain/nnet/normalization.py.
# Modifications by the voxweave authors: only the definitions listed below are
# kept and copied verbatim, except that the definitions from three modules share this
# one file.
"""nnet"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


class Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    groups : int
        Number of blocked connections from input channels to output channels.
    bias : bool
        Whether to add a bias term to convolution operation.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference
    conv_init : str
        Weight initialization for the convolution network
    default_padding: str or int
        This sets the default padding mode that will be used by the pytorch Conv1d backend.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        skip_transpose=False,
        weight_norm=False,
        conv_init=None,
        default_padding=0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=default_padding,
            groups=groups,
            bias=bias,
        )

        if conv_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)
        elif conv_init == "zero":
            nn.init.zeros_(self.conv.weight)
        elif conv_init == "normal":
            nn.init.normal_(self.conv.weight, std=1e-6)

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        Returns
        -------
        wx : torch.Tensor
            The convolved outputs.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(self, x, kernel_size: int, dilation: int, stride: int):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.

        Returns
        -------
        x : torch.Tensor
            The padded outputs.
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError(
                "conv1d expects 2d, 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if not self.padding == "valid" and self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int

    Returns
    -------
    padding : int
        The size of the padding to be added
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = (
            math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        )
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.
    skip_transpose : bool
        Whether to skip the transposition.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=False,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.

        Returns
        -------
        x_n : torch.Tensor
            The normalized outputs.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n
