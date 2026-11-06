"""正文式（6）–（8）的有量纲 Navier–Stokes Corrector。

所有物理与采样参数均由使用者填写；Python 的 None 对应 YAML 的 null。
rho：血液密度，kg/m³；mu：恒定有效动力黏度，Pa·s，按正文采用 CFD Carreau 的 mu_inf。
collocation_points：每病例用于物理约束的点数，正整数；根据数据规模与计算资源填写。

coordinate_scale：输入 t,x,y,z 每通道从网络数值到 SI 数值的比例，单位为 s,m,m,m。
field_scale、field_offset：输出 u,v,w,p 从网络数值到 SI 数值的比例和偏移，
对应关系为 field_SI = field_network * field_scale + field_offset。
三者填写四元素列表或当前批次逐病例的 [B,4] 数组。已经采用 SI 单位时，
显式填写恒等变换；采用逐病例 min-max 归一化时，填写当前病例实际采用的范围和最小值，
包含预处理所用的有效分母。病例归一化范围不同，必须随当前病例更新，不能共用一个范围。
坐标原点偏移不影响本模块中的偏导数，因此无需单独提供坐标偏移。

本模块不填写 Re，也不把运动黏度 nu 当作动力黏度；nu 由 mu/rho 导出。
其他 Carreau 参数用于 CFD 参考场，不在 Corrector 内计算剪切率相关黏度。
"""
import math
import numbers
import torch


def _gradient(value, coords):
    if not value.requires_grad:
        return coords * 0
    derivative = torch.autograd.grad(value, coords, create_graph=True,
                                    retain_graph=True, allow_unused=True)[0]
    return coords * 0 if derivative is None else derivative


def point_derivatives(fields, coords, batch, point):
    """返回指定点 d(u,v,w,p)/d(t,x,y,z)，形状 [4,4]，保留参数计算图。"""
    return torch.stack([_gradient(fields[batch, channel, point], coords)[batch, :, point]
                        for channel in range(4)])


def navier_stokes_residual(fields, coords, rho, mu, indices):
    """计算论文式 (6)-(7)，所有输入须采用 SI 单位。

    fields: [B,4,N]，通道顺序为 u,v,w,p，单位分别为 m/s,m/s,m/s,Pa。
    coords: [B,4,N]，通道顺序为 t,x,y,z，单位分别为 s,m,m,m。
        必须在预测前设置 requires_grad=True，且 fields 必须由这些 coords 计算。
        若网络内部归一化，传入恢复 SI 单位的输出，保留完整归一化计算图。
    rho: 正的有限密度，kg/m³；填写与当前数据/CFD 一致的密度。
    mu: 正的有限动力黏度，Pa·s；按正文填写 CFD Carreau 的 mu_inf。
        不要填写运动黏度 nu (m²/s)，也不要填写 Re 或 1/Re。
    indices: 每病例一个非空、不重复的 torch.long 索引集合，范围为 [0,N)。
        集合大小 K 即式 (8) 的约束点数；按实际采样方案和计算资源填写，
        此处的约束点数与每时相输入点数分别设置。

    返回每病例一个 [K,4] 张量，列为 R_u,R_v,R_w,R_c；动量残差单位
    为 Pa/m，连续性残差单位为 1/s。此处动量残差保留式 (6) 的 rho 系数。
    """
    if any(isinstance(value, bool) or not isinstance(value, numbers.Real)
           or not math.isfinite(value) or value <= 0 for value in (rho, mu)):
        raise ValueError('rho (kg/m³) and mu (Pa·s) must be positive')
    if coords.ndim != 3 or coords.shape[1] != 4 or coords.shape != fields.shape:
        raise ValueError('Physical coordinates and predictions must both have shape [B,4,N]')
    if len(indices) != coords.shape[0]:
        raise ValueError('Exactly one collocation index set is required for each case')
    if not coords.requires_grad:
        raise ValueError('Physical coordinates must require gradients before prediction')
    residuals = []
    for b, selected in enumerate(indices):
        selected = torch.as_tensor(selected, device=coords.device)
        if selected.ndim != 1 or selected.dtype != torch.long or selected.numel() == 0:
            raise ValueError('Collocation indices must be a nonempty one-dimensional integer tensor')
        if (selected < 0).any() or (selected >= coords.shape[-1]).any() or selected.unique().numel() != selected.numel():
            raise ValueError('Collocation indices must be distinct valid point indices')
        rows = []
        for point in selected.tolist():
            derivative = point_derivatives(fields, coords, b, point)
            laplacian = torch.stack([
                sum(_gradient(derivative[c, axis], coords)[b, axis, point]
                    for axis in (1, 2, 3)) for c in range(3)])
            velocity = fields[b, :3, point]
            # 式 (6)：时间导数 + (u,v,w)·空间梯度；各行对应一个速度分量。
            material = derivative[:3, 0] + derivative[:3, 1:4] @ velocity
            # R_phi = rho*(d phi/dt + velocity·grad phi) + dp/dxi - mu*laplacian(phi)
            momentum = rho * material + derivative[3, 1:4] - mu * laplacian
            # 式 (7)：du/dx + dv/dy + dw/dz。
            continuity = derivative[0, 1] + derivative[1, 2] + derivative[2, 3]
            rows.append(torch.cat((momentum, continuity.reshape(1))))
        if not rows:
            raise ValueError('At least one collocation point is required per case')
        residuals.append(torch.stack(rows))
    return residuals


class PhysicsCorrector:
    def __init__(self, rho=None, mu=None, collocation_points=None,
                 coordinate_scale=None, field_scale=None, field_offset=None):
        # 各参数含义、单位和填写方式见本文件顶部说明；不内置数值默认值。
        self.rho, self.mu = rho, mu
        self.collocation_points = collocation_points
        self.coordinate_scale = coordinate_scale
        self.field_scale, self.field_offset = field_scale, field_offset

    @staticmethod
    def _vector(value, name, coords, positive):
        if value is None:
            raise ValueError(f'{name} 为 null，请填写当前病例从网络数值到 SI 单位的变换')
        tensor = torch.as_tensor(value, dtype=coords.dtype, device=coords.device).detach()
        if tensor.shape == (4,):
            tensor = tensor.reshape(1, 4, 1)
        elif tensor.ndim == 2 and tensor.shape[1] == 4:
            tensor = tensor.unsqueeze(-1)
        if (tensor.ndim != 3 or tensor.shape[1:] != (4, 1)
                or tensor.shape[0] not in (1, coords.shape[0])):
            raise ValueError(f'{name} 应填写四元素列表或当前批次的 [B,4] 数组')
        if not torch.isfinite(tensor).all() or (positive and (tensor <= 0).any()):
            raise ValueError(f'{name} 必须有限' + ('且严格为正' if positive else ''))
        return tensor

    def __call__(self, model, coords, labels=None):
        # 保留原有 model, coords, labels 调用接口；labels 不参与物理残差。
        if any(isinstance(v, bool) or not isinstance(v, numbers.Real)
               or not math.isfinite(v) or v <= 0 for v in (self.rho, self.mu)):
            raise ValueError('rho 和 mu 为必填正数；请按 kg/m³ 和 Pa·s 填写')
        if coords.ndim == 2 and coords.shape[1] == 4:
            coords = coords.transpose(0, 1).unsqueeze(0)
        if coords.ndim != 3 or coords.shape[1] != 4 or not coords.is_floating_point():
            raise ValueError('coords 必须为浮点 [B,4,N] 或 [N,4]，通道顺序为 t,x,y,z')
        count = self.collocation_points
        if (isinstance(count, bool) or not isinstance(count, int)
                or not 0 < count <= coords.shape[-1]):
            raise ValueError('collocation_points 为必填正整数，且不能超过每病例输入点数')
        scale = self._vector(self.coordinate_scale, 'coordinate_scale', coords, True)
        output_scale = self._vector(self.field_scale, 'field_scale', coords, True)
        output_offset = self._vector(self.field_offset, 'field_offset', coords, False)
        # 将物理坐标作为求导变量；网络仍接收原来的归一化输入，保持预测器不变。
        physical_coords = (coords.detach()*scale).requires_grad_(True)
        with torch.enable_grad():
            predictions = model(physical_coords/scale)
            physical_fields = predictions*output_scale + output_offset
            indices = [torch.randperm(coords.shape[-1], device=coords.device)[:count]
                       for _ in range(coords.shape[0])]
            values = navier_stokes_residual(physical_fields, physical_coords,
                                           self.rho, self.mu, indices)
            # 式（8）：约束点上四项残差平方之和的均值；各病例等权平均。
            return torch.stack([value.square().mean(0).sum() for value in values]).mean()


def get_pinn(model, coords, labels=None, rho=None, mu=None, collocation_points=None,
             coordinate_scale=None, field_scale=None, field_offset=None):
    corrector = PhysicsCorrector(rho=rho, mu=mu, collocation_points=collocation_points,
        coordinate_scale=coordinate_scale, field_scale=field_scale, field_offset=field_offset)
    return corrector(model, coords, labels)
