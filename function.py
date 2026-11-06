import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz, t_values, new_t_values):
    """
    Input:
        radius: Neighborhood search radius
        nsample: Maximum number of sampled points
        xyz: Coordinates of all points [B, N, 3]
        new_xyz: Coordinates of query points [B, S, 3]
        t_values: Timestamps of all points [B, N, 1]
        new_t_values: Timestamps of query points [B, S, 1]

    Return:
        group_idx: Indices of neighboring points [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]

    t_mask = (t_values.unsqueeze(1) == new_t_values.unsqueeze(2)).squeeze(-1)  # [B, S, N]
    sqrdists[~t_mask] = 1e10 

    # find the nearest nsample points
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)  # [B, S, nsample]

    return group_idx


def query_ball_point_spatial(radius, nsample, xyz, new_xyz):
    """
    Neighborhood query without temporal constraints.
    Input:
        radius: Neighborhood search radius
        nsample: Maximum number of sampled points
        xyz: Coordinates of all points [B, N, 3]
        new_xyz: Coordinates of query points [B, S, 3]

    Return:
        group_idx: Indices of neighboring points [B, S, nsample]
    """

    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, :1].repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, t_values, returnfps=False):
    """
    Input:
        npoint: Number of sampled points
        radius: Neighborhood search radius
        nsample: Maximum number of sampled points within each neighborhood
        xyz: Point cloud coordinates [B, N, 3]
        points: Point cloud features [B, N, D] (including timestamp t)
        t_values: Timestamps [B, N, 1]

    Return:
        new_xyz: Coordinates of sampled points [B, npoint, 3]
        new_points: Features of sampled points [B, npoint, nsample, 3+D]
    """

    B, N, C = xyz.shape
    S = npoint


    fps_idx = farthest_point_sample(xyz, npoint)  # [B, S]
    new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]
    new_t = index_points(t_values, fps_idx)  # [B, S, 1]

    # 2. Find neighboring points with temporal constraints
    group_idx = query_ball_point(radius, nsample, xyz, new_xyz, t_values, new_t)  # [B, S, nsample]

    # 3. ectract grouped points
    grouped_xyz = index_points(xyz, group_idx)  # [B, S, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)  # relative position

    if points is not None:
        grouped_points = index_points(points, group_idx)  # [B, S, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, S, nsample, 3+D]
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


#
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: 点云坐标 [B, 3, N]
            points: 点云特征 [B, D, N] (包含时间戳 t)
        Return:
            new_xyz: 采样后的点坐标 [B, 3, S]
            new_points: 采样后的特征 [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        points = points.permute(0, 2, 1)  # [B, N, D]
        # 提取时间戳 t（假设 t 是 points 的第一个特征）
        t_values = points[..., 0:1]  # [B, N, 1]

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points, t_values
            )

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # [B, D', S]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, S]
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            
            last_channel = in_channel     

            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        
        S = self.npoint
        
        t_values = points[..., 0:1] # [B, N, 1]

        fps_idx = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz, fps_idx)
        
        new_t_values = index_points(t_values, fps_idx) # [B, S, 1]

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            
            group_idx = query_ball_point(radius, K, xyz, new_xyz, t_values, new_t_values)
            
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
# TransFlow
class PointTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.q_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.k_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.v_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        
        self.out_conv1 = nn.Conv1d(out_channels, out_channels, 1)
        self.out_layer_norm = nn.LayerNorm(out_channels)
        self.out_relu = nn.ReLU()
        self.out_conv2 = nn.Conv1d(out_channels, out_channels, 1)


    def forward(self, xyz, features):
        """
        param:
            xyz:  [B, N, 3]
            features:[B, C, N]
        return:
             [B, C_out, 1]
        """
        B, C, N = features.shape
        center_point_feature = features[:, :, :1]
        
        q = self.q_conv(center_point_feature)
        k = self.k_conv(features)
        v = self.v_conv(features)

        center_xyz = xyz[:, :1, :].transpose(1, 2)
        relative_xyz = xyz.transpose(1, 2) - center_xyz
        pos_encoding = self.pos_mlp(relative_xyz.unsqueeze(-1)).squeeze(-1)

        k_with_pos = k + pos_encoding
        
        energy = q.transpose(1, 2) @ k_with_pos
        attention_scores = F.softmax(energy / (C**0.5), dim=-1)

        v_with_pos = v + pos_encoding
        weighted_sum_v = v_with_pos @ attention_scores.transpose(1, 2) # [B, C_out, 1]
        

        x = self.out_conv1(weighted_sum_v)
        
  
        x = x.permute(0, 2, 1)
        x = self.out_layer_norm(x)
        x = x.permute(0, 2, 1) # -> [B, C_out, 1]
        
        x = self.out_relu(x)
        out_features = self.out_conv2(x)

        return out_features

# 新增类二：使用Transformer模块的Set Abstraction层
class PointTransformerSALayer(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, out_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        

        self.transformer_block = PointTransformerBlock(in_channels + 3, out_channels)

    def forward(self, xyz, points):
        """
        Input:
            xyz: Point cloud coordinates [B, 3, N]
            points: Point cloud features [B, D, N] (including timestamp t)

        Return:
            new_xyz: Coordinates of sampled points [B, 3, S]
            new_points: Aggregated features of sampled points [B, C_out, S]
        """

        B, C, N = xyz.shape
        S = self.npoint
        D = points.shape[1]

        xyz_permuted = xyz.permute(0, 2, 1) # [B, N, 3]
        points_permuted = points.permute(0, 2, 1) # [B, N, D]

        fps_idx = farthest_point_sample(xyz_permuted, self.npoint) # [B, S]
        new_xyz_permuted = index_points(xyz_permuted, fps_idx) # [B, S, 3]
        new_xyz = new_xyz_permuted.permute(0, 2, 1) # [B, 3, S]
        
   
        group_idx = query_ball_point_spatial(self.radius, self.nsample, xyz_permuted, new_xyz_permuted)
        
        grouped_xyz = index_points(xyz_permuted, group_idx) # [B, S, nsample, 3]
        grouped_points = index_points(points_permuted, group_idx) # [B, S, nsample, D]
        
        grouped_features = torch.cat([grouped_xyz, grouped_points], dim=-1)
        
        B, S, nsample, C_in_D = grouped_features.shape
        reshaped_features = grouped_features.reshape(B * S, nsample, C_in_D)
        
        reshaped_xyz = grouped_xyz.reshape(B * S, nsample, 3)
        
        reshaped_features_permuted = reshaped_features.permute(0, 2, 1)
        
        agg_features = self.transformer_block(reshaped_xyz, reshaped_features_permuted)
        
        new_points_permuted = agg_features.squeeze(-1).reshape(B, S, -1)
        
        new_points = new_points_permuted.permute(0, 2, 1)
        
        return new_xyz, new_points