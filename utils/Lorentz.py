"""曲率为 -1 的 Lorentz 几何；内部使用 float64，避免时间/空间项相减的精度损失。"""

import torch


MAX_TANGENT_NORM = 5.0


def exp_map(x):
    """原点指数映射：[..., d] -> [..., d+1]；限制半径以防双曲函数溢出。"""
    x = x.to(torch.float64)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    x = x * (MAX_TANGENT_NORM / norm.clamp_min(MAX_TANGENT_NORM))
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(1e-7)
    spatial = x * (torch.sinh(safe_norm) / safe_norm)
    return torch.cat((torch.cosh(norm), spatial), dim=-1)


def log_map(x):
    """原点对数映射，返回 d 个空间分量；asinh 在原点附近比 acosh 更稳定。"""
    spatial = x[..., 1:].to(torch.float64)
    norm = torch.linalg.vector_norm(spatial, dim=-1, keepdim=True).clamp_min(1e-7)
    return spatial * (torch.asinh(norm) / norm)


def lorentz_centroid(mean):
    """将非负权重的邻居坐标均值归一化到双曲面；全零行（无邻居）回退到原点。

    这是 Lorentz 归一化重心，不是精确测地 Fréchet 均值。
    """
    mean = mean.to(torch.float64)
    spatial = mean[..., 1:]
    squared_norm = mean[..., :1].square() - spatial.square().sum(dim=-1, keepdim=True)
    spatial = spatial / squared_norm.clamp_min(1e-12).sqrt()
    # 从空间坐标重建时间坐标，保证上半双曲面约束。
    time = torch.sqrt(1.0 + spatial.square().sum(dim=-1, keepdim=True))
    return torch.cat((time, spatial), dim=-1)
