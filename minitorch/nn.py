from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")
    # get the output dimensions
    out_h = height // kh
    out_w = width // kw
    # reshape and permute the input tensor to match the desired output shape -> split height dim
    permuted_tensor = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, channel, width, out_h, kh)
    )
    # rearrange the dimensions to match the desired output shape
    rearranged_tensor = (
        permuted_tensor.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channel, out_h, out_w, kh * kw)
    )
    return rearranged_tensor, out_h, out_w


# TODO: Implement for Task 4.3.
def avgpool2d(x: Tensor, pool_size: Tuple[int, int]) -> Tensor:
    """2D Average Pooling Operation

    Args:
    ----
        x: Input tensor with shape (batch_size, channels, height, width)
        pool_size: Tuple of (pool_height, pool_width)

    Returns:
    -------
        Pooled output tensor

    """
    # get the input dimensions
    n_batch, n_channels, _, _ = x.shape
    # transform the input into tiled representation
    tiled_tensor, _, _ = tile(x, pool_size)
    # apply the average pooling operation -> compute the mean over the last dimension
    pooled_tensor = tiled_tensor.mean(dim=4)
    pooled_h = tiled_tensor.shape[2]
    pooled_w = tiled_tensor.shape[3]
    return pooled_tensor.view(n_batch, n_channels, pooled_h, pooled_w)


# 4.4
max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input: Input tensor
        dim: Dimension to apply argmax

    Returns:
    -------
        Tensor: Tensor with 1 on highest cell in dim, 0 otherwise

    """
    max_val = max_reduce(input, dim)
    one_hot_t = input == max_val
    return one_hot_t


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:  # noqa: D102
        ctx.save_for_backward(input, int(dim.item()))
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:  # noqa: D102
        saved_input, dim = ctx.saved_values
        one_hot_mask = argmax(saved_input, dim)
        sum_of_mask = one_hot_mask.sum(dim=dim)
        sum_of_mask = sum_of_mask + (sum_of_mask == 0)
        grad_mask = one_hot_mask / sum_of_mask
        return grad_mask * grad_output, 0.0


def max(input_tensor: Tensor, dim: int) -> Tensor:  # noqa: D103
    return Max.apply(input_tensor, input_tensor._ensure_tensor(dim))


def softmax(tensor_input: Tensor, dim: int) -> Tensor:
    r"""Compute the softmax as a tensor.

    Args:
    ----
        tensor_input: Input tensor
        dim: Dimension to apply softmax

    Returns:
    -------
        Tensor: Softmax output tensor

    """
    # subtract the max value
    max_val = max(tensor_input, dim)
    adjusted_input = tensor_input - max_val
    # compute the exponential of the adjusted input
    exp_val = adjusted_input.exp()
    softmax_out = exp_val / exp_val.sum(dim=dim)
    return softmax_out


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of the softmax as a tensor.

    Args:
    ----
        input: Input tensor
        dim: Dimension to apply logsoftmax

    Returns:
    -------
        Tensor: Logsoftmax output tensor

    """
    # Apply the log-sum-exp
    max_val = max(input, dim)
    adjusted_input = input - max_val
    sum_exp = adjusted_input.exp().sum(dim=dim)
    log_sum_exp = sum_exp.log()
    log_softmax_out = adjusted_input - log_sum_exp
    return log_softmax_out


def maxpool2d(input: Tensor, pool_kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: Input tensor
        pool_kernel: Height x width of pooling

    Returns:
    -------
        Tensor: Pooled output tensor

    """
    num_batches, num_channels, _, _ = input.shape
    # reshape the input tensor for pooling
    reshaped_t, pool_h, pool_w = tile(input, pool_kernel)
    # apply max over the last dim
    pooled_t = max(reshaped_t, dim=4).view(num_batches, num_channels, pool_h, pool_w)
    return pooled_t


def dropout(input: Tensor, prob: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: Input tensor
        prob: Probability of dropping out each position
        ignore: Skip dropout, i.e. do nothing

    Returns:
    -------
        Tensor: Tensor with random positions dropped out

    """
    if ignore:
        return input
    else:
        # generate random noise with the same shape as input
        noise_t = rand(input.shape)
        # create a mask where elements are true if rand > prob
        mask_bool = noise_t > prob
        # apply the mask to input tensor
        return input * mask_bool
