# -*- coding: utf-8 -*-
"""
Ref:
    - https://github.com/mir-group/nequip
    - https://www.nature.com/articles/s41467-022-29939-5
"""

import math
from typing import Optional

import torch
from e3nn.math import soft_one_hot_linspace
from torch import nn

from mattersim.jit_compile_tools.jit import compile_mode


class e3nn_basias(nn.Module):
    def __init__(
        self,
        r_max: float,
        r_min: Optional[float] = None,
        e3nn_basis_name: str = "gaussian",
        num_basis: int = 8,
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min if r_min is not None else 0.0
        self.e3nn_basis_name = e3nn_basis_name
        self.num_basis = num_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return soft_one_hot_linspace(
            x,
            start=self.r_min,
            end=self.r_max,
            number=self.num_basis,
            basis=self.e3nn_basis_name,
            cutoff=True,
        )

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn(5, 1),)} for _ in range(n)]


class BesselBasis(nn.Module):
    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in
            DimeNet: https://arxiv.org/abs/2003.03123

        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(
            self.bessel_weights * x.unsqueeze(-1) / self.r_max  # noqa: E501
        )

        return self.prefactor * (numerator / x.unsqueeze(-1))


@compile_mode("script")
class SmoothBesselBasis(nn.Module):
    def __init__(self, r_max, max_n=10):
        r"""Smooth Radial Bessel Basis, as proposed
            in DimeNet: https://arxiv.org/abs/2003.03123
            This is an orthogonal basis with first
            and second derivative at the cutoff
            equals to zero. The function was derived from
            the order 0 spherical Bessel function,
            and was expanded by the different zero roots
        Ref:
            https://arxiv.org/pdf/1907.02374.pdf
        Args:
            r_max: torch.Tensor distance tensor
            max_n: int, max number of basis, expanded by the zero roots
        Returns: expanded spherical harmonics with
                 derivatives smooth at boundary
        Parameters
        ----------
        """
        super(SmoothBesselBasis, self).__init__()
        self.max_n = max_n
        n = torch.arange(0, max_n).float()[None, :]
        PI = 3.1415926535897
        SQRT2 = 1.41421356237
        fnr = (
            (-1) ** n
            * SQRT2
            * PI
            / r_max**1.5
            * (n + 1)
            * (n + 2)
            / torch.sqrt(2 * n**2 + 6 * n + 5)
        )
        en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = [torch.tensor(1.0).float()]
        for i in range(1, max_n):
            dn.append(1 - en[0, i] / dn[-1])
        dn = torch.stack(dn)
        self.register_buffer("dn", dn)
        self.register_buffer("en", en)
        self.register_buffer("fnr_weights", fnr)
        self.register_buffer(
            "n_1_pi_cutoff",
            ((torch.arange(0, max_n).float() + 1) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer(
            "n_2_pi_cutoff",
            ((torch.arange(0, max_n).float() + 2) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer("r_max", torch.tensor(r_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Smooth Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        x_1 = x.unsqueeze(-1) * self.n_1_pi_cutoff
        x_2 = x.unsqueeze(-1) * self.n_2_pi_cutoff
        fnr = self.fnr_weights * (torch.sin(x_1) / x_1 + torch.sin(x_2) / x_2)
        gn = [fnr[:, 0]]
        for i in range(1, self.max_n):
            gn.append(
                1
                / torch.sqrt(self.dn[i])
                * (
                    fnr[:, i]
                    + torch.sqrt(self.en[0, i] / self.dn[i - 1]) * gn[-1]  # noqa: E501
                )
            )
        return torch.transpose(torch.stack(gn), 1, 0)


class WeightedFeatureFusion(nn.Module):
    """
    融合 SmoothBesselBasis 与多种经典势函数：
    Lennard-Jones, Morse, Buckingham, Coulomb。
    输出与 Bessel Basis 一致 [num_edges, max_n]。
    """
    def __init__(
        self,
        r_max: float,
        max_n: int = 10,
        include_LJ: bool = True,
        include_Morse: bool = True,
        include_Buck: bool = True,
        include_Coul: bool = True,
        # charges: Tensor = None,  # 必须提供原子电荷，用于 Coulomb 势
    ):
        super().__init__()
        self.r_max = r_max
        self.max_n = max_n
        self.bessel = SmoothBesselBasis(r_max=r_max, max_n=max_n)
        self.include_LJ = include_LJ
        self.include_Morse = include_Morse
        self.include_Buck = include_Buck
        self.include_Coul = include_Coul
        # self.charges = charges  # [num_atoms] 或 [num_edges, 2]

        # 可训练权重
        self.w_bessel = nn.Parameter(torch.ones(max_n))
        if include_LJ:
            self.w_LJ = nn.Parameter(torch.ones(1))
        if include_Morse:
            self.w_Morse = nn.Parameter(torch.ones(1))
        if include_Buck:
            self.w_Buck = nn.Parameter(torch.ones(1))
        if include_Coul:
            self.w_Coul = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [num_edges] 距离张量
        输出: [num_edges, max_n]
        """
        # --- Bessel ---
        R_bessel = self.bessel(x)  # [num_edges, max_n]
        fused = R_bessel * self.w_bessel  # 广播权重

        # --- Lennard-Jones ---
        if self.include_LJ:
            epsilon, sigma = 1.0, 1.0
            lj = 4 * epsilon * ((sigma / x) ** 12 - (sigma / x) ** 6)
            lj_exp = lj.unsqueeze(-1).repeat(1, self.max_n)
            fused = fused + self.w_LJ * lj_exp

        # --- Morse ---
        if self.include_Morse:
            D, alpha, r0 = 1.0, 1.0, 1.0
            morse = D * (1 - torch.exp(-alpha * (x - r0))) ** 2
            morse_exp = morse.unsqueeze(-1).repeat(1, self.max_n)
            fused = fused + self.w_Morse * morse_exp

        # --- Buckingham ---
        if self.include_Buck:
            A, B, C = 1.0, 1.0, 1.0
            buck = A * torch.exp(-B * x) - C / x ** 6
            buck_exp = buck.unsqueeze(-1).repeat(1, self.max_n)
            fused = fused + self.w_Buck * buck_exp

        # --- Coulomb ---
        if self.include_Coul:
            # if self.charges is None:
            #     raise ValueError("Coulomb 势需要提供 charges 张量")
            # # 假设 charges 为 [num_edges, 2]
            # q1, q2 = self.charges[:, 0], self.charges[:, 1]
            # coul = q1 * q2 / x  # 可加入 1/(4πε0) 系数
            coul = 1 / x  # 可加入 1/(4πε0) 系数
            coul_exp = coul.unsqueeze(-1).repeat(1, self.max_n)
            fused = fused + self.w_Coul * coul_exp

        return fused  # [num_edges, max_n]

# import torch
# def test_smooth_bessel_and_weighted_fusion():
#     # === 测试参数 ===
#     num_edges = 5
#     r_max = 5.0
#     max_n = 8

#     # 随机生成距离张量 [num_edges]
#     torch.manual_seed(0)
#     edge_length = torch.rand(num_edges) * r_max  # 范围 0~r_max

#     print("=== edge_length ===")
#     print(edge_length)

#     # === 测试 SmoothBesselBasis ===
#     bessel = SmoothBesselBasis(r_max=r_max, max_n=max_n)
#     bessel_out = bessel(edge_length)

#     print("\n=== SmoothBesselBasis 输出 ===")
#     print("Shape:", bessel_out.shape)  # 期望 [num_edges, max_n]
#     print(bessel_out)

#     # === 测试 WeightedFeatureFusion ===
#     fusion = WeightedFeatureFusion(
#         r_max=r_max, 
#         max_n=max_n, 
#         include_LJ=True, 
#         include_Morse=True, 
#         include_Buck=True, 
#         include_Coul=True
#     )
#     fusion_out = fusion(edge_length)

#     print("\n=== WeightedFeatureFusion 输出 ===")
#     print("Shape:", fusion_out.shape)  # 期望 [num_edges, max_n]
#     print(fusion_out)

#     # === 检查输出是否存在 NaN 或 Inf ===
#     assert not torch.isnan(bessel_out).any(), "SmoothBesselBasis 输出包含 NaN!"
#     assert not torch.isinf(bessel_out).any(), "SmoothBesselBasis 输出包含 Inf!"
#     assert not torch.isnan(fusion_out).any(), "WeightedFeatureFusion 输出包含 NaN!"
#     assert not torch.isinf(fusion_out).any(), "WeightedFeatureFusion 输出包含 Inf!"

#     print("\n测试通过 ✅ SmoothBesselBasis 和 WeightedFeatureFusion 输出正常。")

# if __name__ == "__main__":
#     test_smooth_bessel_and_weighted_fusion()


# class GaussianBasis(nn.Module):
#     r_max: float

#     def __init__(self, r_max, r_min=0.0, num_basis=8, trainable=True):
#         super().__init__()

#         self.trainable = trainable
#         self.num_basis = num_basis

#         self.r_max = float(r_max)
#         self.r_min = float(r_min)

#         means = torch.linsspace(self.r_min, self.r_max, self.num_basis)
#         stds = torch.full(size=means.size, fill_value=means[1] - means[0])
#         if self.trainable:
#             self.means = nn.Parameter(means)
#             self.stds = nn.Parameter(stds)
#         else:
#             self.register_buffer("means", means)
#             self.register_buffer("stds", stds)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = (x[..., None] - self.means) / self.stds
#         x = x.square().mul(-0.5).exp() / self.stds  # sqrt(2 * pi)
