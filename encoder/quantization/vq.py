# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Residual vector quantizer implementation."""

from dataclasses import dataclass, field
import math
import typing as tp

import torch
from torch import nn

from encoder.quantization.core_vq import ResidualVectorQuantization,LanguageVectorQuantization
from encoder.quantization.ddp_core_vq import DistributedMScaleRVQ


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # print(self.bins)

        # breakpoint()

        self.vq = LanguageVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )
        # self.vq = ResidualVectorQuantization(
        #     dim=self.dimension,
        #     codebook_size=self.bins,
        #     num_quantizers=self.n_q,
        #     decay=self.decay,
        #     kmeans_init=self.kmeans_init,
        #     kmeans_iters=self.kmeans_iters,
        #     threshold_ema_dead_code=self.threshold_ema_dead_code,
        # )


    def forward(self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            frame_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        # breakpoint()


        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        # assert n_q==4
        # breakpoint()
        # nq_choice=[3,4,8]
        nq_choice=[4,6,8]
        if self.training:
            # choice = int(torch.randint(0, 3, (1,)).item())
            choice = int(torch.randint(0, 3, (1,)).item())
        # breakpoint()
            n_q=nq_choice[choice]
        # breakpoint()
        # n_q=8
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))

    def infer(self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            frame_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        # n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        # # assert n_q==4
        # # breakpoint()
        # # nq_choice=[3,4,8]
        # nq_choice=[3,4,5,6,7,8]
        # if self.training:
        #     # choice = int(torch.randint(0, 3, (1,)).item())
        #     choice = int(torch.randint(0, 6, (1,)).item())
        # # breakpoint()
        #     n_q=nq_choice[choice]
        n_q=1
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))

    def get_num_quantizers_for_bandwidth(self, frame_rate: int, bandwidth: tp.Optional[float] = None) -> int:
        """Return n_q based on specified target bandwidth.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.:
            # bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented as
            # bandwidth == 6.0
            n_q = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, frame_rate: int):
        """Return bandwidth per quantizer for a given input frame rate.
        Each quantizer encodes a frame with lg(bins) bits.
        """
        return math.log2(self.bins) * frame_rate

    def encode(self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizers to use
        and returns indices for each quantizer.
        """
        n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        codes = self.vq.encode(x, n_q=n_q)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        """
        quantized = self.vq.decode(codes)
        return quantized



class MScaleRVQ(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        quantize_dropout: bool = False,
        rand_num_quant: tp.Optional[tp.List] = None,
        encoder_hop_length: int = 320,
        use_ddp: bool = True,
        q0_ds_ratio: int = 1,
        vq_scales: tp.Optional[tp.List[int]] = None
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.encoder_hop_length = encoder_hop_length
        self.training = True
        self.vq_scales = vq_scales
        # rvq_class = ResidualVectorQuantization
        # if use_ddp:
        #     rvq_class = DistributedResidualVectorQuantization
        #     logging.info("Using distributed residual vector quantization.")
        # else:
        #     logging.warning("ResidualVectorQuantization will be removed in the future version, set use_ddp=True.")

        self.model = DistributedMScaleRVQ(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            quantize_dropout=quantize_dropout,
            rand_num_quant=rand_num_quant,
            q0_ds_ratio=q0_ds_ratio,
            vq_scales=vq_scales
        )

    def forward(self, x: torch.Tensor, sample_rate: int, n_q: tp.Optional[int] = None, rand_quantize_dropout_index=None) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor in the shape of (B, C, T).
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        if n_q is None:
            n_q = self.n_q
        quantized, codes, commit_loss, sub_quants = self.model(x, n_q=n_q, rand_quantize_dropout_index=rand_quantize_dropout_index)
        bw = torch.tensor(sum([sample_rate // self.encoder_hop_length // self.vq_scales[i] for i in range(n_q)]))
        return QuantizedResult(quantized, codes, bw,
                               penalty=torch.mean(commit_loss))

    def get_num_quantizers_for_bandwidth(self, sample_rate: int, bandwidth: tp.Optional[float] = None) -> int:
        """Return n_q based on specified target bandwidth.
        """
        bw = 0
        n_q = 1
        for i in range(self.n_q):
            bw += sample_rate // self.encoder_hop_length // self.vq_scales[i]
            if bw > bandwidth:
                break
            n_q = i + 1
        return n_q

    def encode(self, x: torch.Tensor, sample_rate: int, n_q: tp.Optional[float] = None) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        codes = self.model.encode(x, n_q=n_q)
        return codes

    def decode(self, codes: tp.List[int]) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        """
        quantized = self.model.decode(codes)
        return quantized


if __name__ == "__main__":
    mscale_rvq = MScaleRVQ(
        dimension=256,
        n_q=4,
        vq_scales=[8, 4, 2, 1]
    )
    # x = torch.randn(1, 256, 10000)
    # quantized, codes, commit_loss, sub_quants = mscale_rvq(x, 16000, n_q=8)
    # print(quantized.shape, codes.shape, commit_loss.shape, sub_quants.shape)
    total = sum([param.nelement() for param in mscale_rvq.parameters()])
    print(f"Number of parameter: {total/1e6:.2f}M")
    x = torch.randn(1, 256, 64)

    quantized_result1 = mscale_rvq(x, 16000, n_q=4)
    quantized_result2 = mscale_rvq(x, 16000, n_q=4)
    codes = mscale_rvq.encode(x, sample_rate=16000, n_q=4)

    # print(quantized_result.quantized.shape)
    print(quantized_result1.codes)
    print(codes)
    quantized = mscale_rvq.decode(codes)
    # print(quantized_result.bandwidth)
    # print(quantized_result.penalty)
    # print(quantized_result.sub_quants)

    for c1, c2 in zip(codes, quantized_result1.codes):
        print(torch.allclose(c1, c2))
        print(c1.shape)

    # print(torch.allclose(quantized, quantized_result.quantized, atol=1e-4))
    # print(quantized)
    # print(quantized_result.quantized)
    print(torch.allclose(quantized_result2.quantized, quantized_result1.quantized, atol=1e-5))

    total = sum([param.nelement() for param in mscale_rvq.parameters()])
    print(f"Number of parameter: {total/1e6:.2f}M")


    rvquantizer = ResidualVectorQuantizer()
    total = sum([param.nelement() for param in rvquantizer.parameters()])
    print(f"Number of parameter: {total/1e6:.2f}M")