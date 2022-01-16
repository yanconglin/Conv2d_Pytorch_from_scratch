# torch.nn.Conv2d from scratch, based on Deformable-ConvNets-V2

Many thanks to the authors/developers from [mmdetection branch](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch).

## Test

```
python test.py (tested with Pytorch 1.7.0)
```

See `test.py` for example usage, inculuding: forward pass, gradient check, and timing.

Comparison with nn.Conv2d on a 16x128x128x128 input (in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)

Average time over 10 runs: Pytorch (us): 5800.2046

Average time over 10 runs: deformconv (us): 20601.4746

