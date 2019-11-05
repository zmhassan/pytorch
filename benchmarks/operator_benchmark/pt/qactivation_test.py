from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

import torch
import torch.quantization as tq
import torch.nn.quantized as nnq

import operator_benchmark as op_bench

r"""Microbenchmarks for the quantized activations."""

qrelu_configs = op_bench.cross_product_configs(
    dims=(
        (1,), (1, 1), (1, 1, 1),     # Single element
        (2, 1), (1, 2),              # Rank=2 row-/col-major
        (3, 4, 5),                   # Rank=3
        (1, 3, 4, 5), (2, 3, 4, 5),  # Rank=4, batch=1, batch>1
        (4, 1, 1, 1),                # Rank=4, all other single dimensions
    ),
    permute_dims=(False, True),
    inplace=(False, True),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    tags=('short',)
)


class QReLUBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims, permute_dims, inplace, dtype):
        self.qop = nnq.ReLU(inplace=inplace)

        # Input dimensions
        f_input = (torch.rand(*dims) - 0.5) * 1e6

        # Get quantization paramerters and quantize
        if dtype in (torch.qint8, torch.quint8):
            observer = tq.MinMaxObserver(dtype=dtype,
                                         qscheme=torch.per_tensor_affine,
                                         reduce_range=False)
            observer.forward(f_input)
            scale, zero_point = observer.calculate_qparams()
            scale, zero_point = scale.item(), zero_point.item()
        else:
            zero_point = 0
            qinfo = torch.iinfo(dtype)
            fmin, fmax = f_input.min(), f_input.max()
            scale = (fmax - fmin).item() / (qinfo.max - qinfo.min)

        # Quantize the tensor
        self.q_input = torch.quantize_per_tensor(f_input, scale=scale,
                                                 zero_point=zero_point,
                                                 dtype=dtype)
        if permute_dims:
            # Make non-contiguous
            dims = list(dims)
            random.shuffle(dims)
            self.q_input = self.q_input.permute(dims)

        self.set_module_name("QReLU")

    def forward(self):
        return self.qop(self.q_input)

op_bench.generate_pt_test(qrelu_configs, QReLUBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
