#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from deform_conv import DeformConv, DeformConvFunction
import torch.utils.benchmark as benchmark
from timer import PyTorchTimer

batch=16
inC, outC, inH, inW = 128, 128, 128, 128
kernel = 3
stride=1
padding=1
dilation=1
groups=1
bias=True
im2col_step=64

torch.manual_seed(2022)

### pytorch conv2d
Conv2d = nn.Conv2d(inC, outC, kernel, stride, padding, dilation, groups, bias).cuda()

### deformable conv2d
### why this? https://github.com/pytorch/pytorch/issues/16940
def Conv2d_deformable_func(input, weight, bias, stride, padding, dilation, groups, im2col_step):
    return DeformConvFunction.apply(input, weight, bias, stride, padding, dilation, groups, im2col_step)

Conv2d_deformable = DeformConv(inC, outC, kernel, stride, padding, dilation, groups, bias).cuda()

########################
def run_conv():
    input = torch.randn(batch, inC, inH, inW).cuda()
    out = Conv2d(input)
    print('Conv2d_pytorch, input & output shape: ', input.data.shape, out.data.shape)

def run_deformconv():
    input = torch.randn(batch, inC, inH, inW).cuda()
    out = Conv2d_deformable(input)
    print('Conv2d_deform, input & output shape: ', input.data.shape, out.data.shape)

def time_comparison():
    t0 = benchmark.Timer(
        stmt='run_conv()',
        setup='from __main__ import run_conv',
        globals={'x': x})

    t1 = benchmark.Timer(
        stmt='run deformconv()',
        setup='from __main__ import run',
        globals={'x': x})

    print(t0.timeit(100))
    print(t1.timeit(100))
def run_comparison():
    weight = nn.Parameter(torch.randn(outC, inC//groups, kernel, kernel).double()).cuda()
    bias = nn.Parameter(torch.randn(outC).double()).cuda()
    # print("weight", weight.shape, weight)
    # print("bias", bias.shape, bias)

    input = torch.randn(batch, inC, inH, inW).double().cuda()

    out1 = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    out2 = Conv2d_deformable_func(input, weight, bias, stride, padding, dilation, groups, im2col_step)
    d = (out1 - out2).abs().max()

    print('Maximal error (abs): {}'.format(d))



def check_gradient_deformconv():

    input = torch.rand(batch, inC, inH, inW).double().cuda()
    input.requires_grad = True

    weight = torch.randn(outC, int(inC//groups), kernel, kernel).double().cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).double().cuda()
    bias.requires_grad = True

    # print('check_gradient_deformconv: ',
    #       gradcheck(DeformConvFunction, (input, weight, bias,
    #                 stride, padding, dilation, groups, im2col_step),
    #                 eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True))
    print('check_gradient_deformconv: ',
          gradcheck(DeformConvFunction.apply, (input, weight, bias,
                    stride, padding, dilation, groups, im2col_step), raise_exception=True))


if __name__ == '__main__':
    
    print("Pytorch version: ", torch.__version__)
    print("GPU version: ", torch.cuda.get_device_name())
    
    run_conv()
    run_deformconv()
    run_comparison()
    # check_gradient_deformconv()

    max_num=10
    timer_pytorch = PyTorchTimer(cuda=True, name="Pytorch", keep_n=max_num)
    for k in range(max_num):
        input = torch.randn(batch, inC, inH, inW).cuda()
        timer_pytorch.start()
        out = Conv2d(input)
        # out = Conv2d_deformable(input)
        timer_pytorch.stop()
    print(timer_pytorch.__str__())

    time_deformconv = PyTorchTimer(cuda=True, name="deformconv", keep_n=max_num)
    for k in range(max_num):
        input = torch.randn(batch, inC, inH, inW).cuda()
        time_deformconv.start()
        # out = Conv2d(input)
        out = Conv2d_deformable(input)
        time_deformconv.stop()
    print(time_deformconv.__str__())




    # """
    # ****** Note: backward is not reentrant error may not be a serious problem,
    # ****** since the max error is less than 1e-7,
    # ****** Still looking for what trigger this problem
    # """
