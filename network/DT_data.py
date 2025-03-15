import numpy as np

import torch

from scipy import sparse as sp

import os
import torch.nn.functional as f
import open3d
from time import *
from scipy.spatial import cKDTree
import math


class ScanData:
    def __init__(self):
        self.pc = None
        self.surface_pc = None
        self.scan_name = None
        self.data_para = None
        self.cell_vertex_idx = None
        self.adj = None
        self.ref_label = None
        self.output_facet_vertex_idx = None
        self.output_facet_nei_cell = None
        self.tet_adj_facet = None

    def load_full_scan(self, file_path, cfg):
        print('file_path', file_path)
        self.pc = open3d.io.read_point_cloud(os.path.join(file_path, "sampled_points.ply"))
        # self.surface_pc = open3d.io.read_point_cloud(os.path.join(file_path, "gt_surface_sample_points.ply"))

        file_names = dict()
        file_names['output_tetrahedron_adj'] = (os.path.join(file_path, "output_tetrahedron_adj"))
        file_names['output_cell_vertex_idx'] = (os.path.join(file_path, "output_cell_vertex_idx.txt"))
        file_names['ref_label'] = (os.path.join(file_path, "ref_point_label.txt"))
        file_names['output_facet_vertex_idx'] = (os.path.join(file_path, "output_facet_vertex_idx.txt"))
        file_names['output_facet_nei_cell'] = (os.path.join(file_path, "output_facet_nei_cell.txt"))
        file_names['tet_adj_facet'] = (os.path.join(file_path, "tet_adj_facet.txt"))

        l = dict()
        l['ref_label'] = np.fromfile(file_names['ref_label'], dtype=np.int32, sep=' ').reshape(-1, cfg["ref_num"])
        l['adj_mat'] = np.fromfile(file_names['output_tetrahedron_adj'], dtype=np.int32, sep=' ').reshape(-1, 4)
        l['cell_vertex_idx'] = np.fromfile(file_names['output_cell_vertex_idx'], dtype=np.int32, sep=' ').reshape(-1, 4)
        l['output_facet_vertex_idx'] = np.fromfile(file_names['output_facet_vertex_idx'], dtype=np.int32,sep=' ').reshape(-1, 3)
        l['output_facet_nei_cell'] = np.fromfile(file_names['output_facet_nei_cell'], dtype=np.int32, sep=' ').reshape(-1, 2)
        l['tet_adj_facet'] = np.fromfile(file_names['tet_adj_facet'], dtype=np.int32, sep=' ').reshape(-1, 4)

        self.output_facet_vertex_idx = l['output_facet_vertex_idx']
        self.output_facet_nei_cell = l['output_facet_nei_cell']
        self.cell_vertex_idx = (l['cell_vertex_idx'])
        self.adj = l['adj_mat']
        self.tet_adj_facet = l['tet_adj_facet']
        self.ref_label = l['ref_label'][:, 0:cfg["ref_num"]]
        print("loaded " + file_path.split('/')[-2])


class DT_Data(object):
    def __init__(self):
        self.label_weights = None
        self.cell_vertex_idx = None
        self.adj = None
        self.adj_idx = None
        self.ref_label = None
        self.facet_vertex_idx = None
        self.facet_nei_cell = None
        self.data_name = ""
        self.neigh_idx = None
        self.tet_adj_facet = None
        self.batch_point_feature = None
        self.batch_cell_feature = None
        self.batch_facet_feature = None
        self.facet_length_max = None
        self.cell_offset_xyz = None
        self.surface_sample_xyz = None
        self.dis_surface_to_delaunay = None
        self.sorted_inds = None
        self.infinite_cell = None

    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def to(self, device):
        device_data = DT_Data()
        self_keys = self.keys()
        for key in self_keys:
            if isinstance(self[key], list):
                for e in self[key]:
                    if torch.is_tensor(e):
                        if e.type() == 'torch.IntTensor':
                            device_data[key].append(e.to(device, dtype=torch.long))
                        else:
                            device_data[key].append(e.to(device))
                    else:
                        device_data[key].append(e)
            elif torch.is_tensor(self[key]):
                if self[key].type() == 'torch.IntTensor':
                    device_data[key] = self[key].to(device, dtype=torch.long)
                else:
                    device_data[key] = self[key].to(device)
        return device_data


def create_full_data(scan, cfg):
    out = DT_Data()
    out.data_name = scan.scan_name
    out.facet_vertex_idx = torch.from_numpy(scan.output_facet_vertex_idx)
    out.facet_nei_cell = torch.from_numpy(scan.output_facet_nei_cell)
    out.tet_adj_facet = torch.from_numpy(scan.tet_adj_facet)

    current_xyz, centroid, m = pc_normalize(np.asarray(scan.pc.points))
    current_xyz = torch.from_numpy(current_xyz).float()
    current_normals = torch.from_numpy(np.asarray(scan.pc.normals)).float()
    current_normals = f.normalize(current_normals, p=2, dim=1)

    neigh_idx = get_neighbor_idx(current_xyz.numpy(), current_xyz.numpy(), cfg["k_n"])
    neigh_idx = neigh_idx.long()
    neigh_normals = current_normals[neigh_idx]
    neighbor_xyz = current_xyz[neigh_idx]
    relative_xyz = get_local_xyz(current_xyz, neighbor_xyz)

    batch_point_feature = torch.cat((relative_xyz, neigh_normals), dim=2)
    out.neigh_idx = neigh_idx  # 邻居点
    out.batch_point_feature = batch_point_feature

    if cfg["use_sample_label"]:
        tmp_labels = np.asarray(scan.ref_label) - 1
        out.ref_label = torch.from_numpy(tmp_labels)
        if scan.data_para is not None:
            out.label_weights = torch.from_numpy(scan.data_para.class_weights)

    out.cell_vertex_idx = torch.from_numpy(np.asarray(scan.cell_vertex_idx))
    out.adj_idx = torch.from_numpy(np.asarray(scan.adj))

    infinite_cell = get_infinite_cell(out.cell_vertex_idx)
    out.infinite_cell = infinite_cell
    cell_vertex_xzy = current_xyz[out.cell_vertex_idx.long()]
    cell_center_xyz = torch.mean(cell_vertex_xzy, dim=1)
    cell_offset_xyz = get_local_xyz(cell_center_xyz, cell_vertex_xzy)
    cell_offset_xyz[out.infinite_cell] = 0.
    out.cell_offset_xyz = cell_offset_xyz

    batch_cell_feature, cell_ball_centers = get_cell_feature(cell_vertex_xzy)
    batch_cell_feature[out.infinite_cell] = 0.
    out.batch_cell_feature = batch_cell_feature

    facet_feature, length_max = get_facet_feature(out.facet_vertex_idx, current_xyz,out.facet_nei_cell, infinite_cell, cell_ball_centers)
    out.batch_facet_feature = facet_feature
    length_max = length_max.squeeze(1)
    out.facet_length_max = length_max

    return out


def get_neighbor_idx_noself(pc, query_pts, k):
    kdtree = cKDTree(pc)
    (x, idx) = kdtree.query(query_pts, k + 1)
    idx = idx[:, 1:]
    idx = torch.from_numpy(idx)

    return idx


def get_neighbor_idx(pc, query_pts, k):
    kdtree = cKDTree(pc)
    (x, idx) = kdtree.query(query_pts, k)
    idx = torch.from_numpy(idx)

    return idx


def get_depth(point_xyz, neighbor_normals, neighbor_xyz, k):
    relative_xyz = point_xyz.unsqueeze(1) - neighbor_xyz
    depth = torch.mul(neighbor_normals, relative_xyz)
    depth = torch.sum(depth, 2)

    return depth


def get_normal(point_normals, neighbor_normals):
    print("point_normals.size()", point_normals.size())
    depth = torch.sum(torch.mul(neighbor_normals, point_normals.unsqueeze(1)), 2, keepdim=True)
    normals1 = torch.mul(depth, neighbor_normals)
    normals2 = point_normals.unsqueeze(1) - normals1
    relative_normals = torch.cat((normals1, normals2), dim=2)
    print("relative_normals.size()", relative_normals.size())

    return relative_normals


def get_local_xyz(current_xyz, neighbor_xyz):
    current_xyz = current_xyz.unsqueeze(1)
    local_xyz = current_xyz - neighbor_xyz

    return local_xyz


def pc_normalize(point_xyz):

    centroid = np.mean(point_xyz, axis=0)
    point_xyz = point_xyz - centroid
    m = np.max(np.sqrt(np.sum(point_xyz ** 2, axis=1)))
    point_xyz = point_xyz / m

    return point_xyz, centroid, m


def surface_pc_normalize(surface_pc, centroid, m):
    point_xyz = np.asarray(surface_pc.points)
    point_xyz = point_xyz - centroid
    point_xyz = point_xyz / m
    point_xyz = torch.from_numpy(point_xyz).float()
    point_normals = torch.from_numpy(np.asarray(surface_pc.normals)).float()
    point_normals = f.normalize(point_normals, p=2, dim=1)

    return point_xyz, point_normals


def depth_normalize(depth):
    max_depth = torch.clamp(torch.max(torch.abs(depth), 1, keepdim=True)[0], min=1e-12)
    depth = depth / max_depth

    return depth


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = xyz.unsqueeze(0)
    # device = torch.device("cuda:{}".format(cfg["device_ids"][0]))
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1).to(device)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    centroids = centroids.cpu().squeeze(0)
    torch.cuda.empty_cache()

    return centroids


def point_triangle_distance(surface_sample_xyz, current_xyz, facet_vertex_idx):
    facet_vertex_xyz = current_xyz[facet_vertex_idx.long()]
    facet_center = torch.mean(facet_vertex_xyz, dim=1)
    facet_vertex_a = facet_vertex_xyz[:, 0, :]
    facet_vertex_b = facet_vertex_xyz[:, 1, :]
    facet_vertex_c = facet_vertex_xyz[:, 2, :]
    ab = facet_vertex_b - facet_vertex_a
    ca = facet_vertex_a - facet_vertex_c
    fn = f.normalize(torch.cross(ab, -1.0 * ca, dim=1), p=2, dim=-1)
    surface_sample_xyz = surface_sample_xyz.unsqueeze(1).expand(-1, facet_center.shape[0], -1)
    facet_center = facet_center.unsqueeze(0).expand(surface_sample_xyz.shape[0], -1, -1)
    distance = surface_sample_xyz - facet_center
    fn = fn.unsqueeze(0).expand(surface_sample_xyz.shape[0], -1, -1)
    SDF = torch.abs(torch.sum(torch.mul(distance, fn), dim=2))
    k_val = min(32, SDF.shape[-1])
    SDF, sorted_inds = torch.topk(SDF, largest=False, k=k_val, dim=-1)
    SDF = 1e+5 * SDF

    return SDF, sorted_inds


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_infinite_cell(cell_vertex_idx):
    infinite_cell = []
    for i in range(cell_vertex_idx.shape[0]):
        if -1 in cell_vertex_idx[i]:
            infinite_cell.append(i)
    infinite_cell = torch.from_numpy(np.array(infinite_cell)).long()

    return infinite_cell


def get_cell_feature(cell_vertex_xzy):
    cell_side_length = []
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            side_length = cell_vertex_xzy[:, i, :] - cell_vertex_xzy[:, j, :]
            side_length = torch.norm(side_length, dim=1, keepdim=True).numpy()
            cell_side_length.append(side_length)
    cell_side_length = torch.from_numpy(np.array(cell_side_length)).transpose(0, 2).squeeze(0)
    length_max = torch.max(cell_side_length, dim=1, keepdim=True)[0]
    length_min = torch.min(cell_side_length, dim=1, keepdim=True)[0]
    cell_side = torch.cat((length_max, length_min), dim=1)

    side_ab = cell_vertex_xzy[:, 0, :] - cell_vertex_xzy[:, 1, :]
    side_ac = cell_vertex_xzy[:, 0, :] - cell_vertex_xzy[:, 2, :]
    side_ad = cell_vertex_xzy[:, 0, :] - cell_vertex_xzy[:, 3, :]
    cell_volume = torch.abs(torch.sum(torch.mul(torch.cross(side_ab, side_ac), side_ad), dim=1, keepdim=True)) / 6.0

    point_a = cell_vertex_xzy[:, 0]
    point_b = cell_vertex_xzy[:, 1]
    point_c = cell_vertex_xzy[:, 2]
    point_d = cell_vertex_xzy[:, 3]
    circle_center, r = round_computation(point_a, point_b, point_c)
    ab = point_b - point_a
    bc = point_c - point_b
    ca = point_a - point_c
    fn = f.normalize(torch.cross(ab, -1.0 * ca, dim=1), p=2, dim=-1)
    m = torch.sum(torch.pow((circle_center - point_a), 2), dim=1, keepdim=True) - \
        torch.sum(torch.pow((circle_center - point_d), 2), dim=1, keepdim=True)
    n = torch.sum(fn * (point_a - point_d), dim=1, keepdim=True) * 2.0
    k = m / n
    cell_ball_center = circle_center + fn * k
    cell_ball_radius = torch.norm((cell_ball_center - point_a), dim=1, keepdim=True)
    cell_ball_radius = torch.clamp(cell_ball_radius, max=30000.)

    cell_feature = torch.cat((cell_side, cell_ball_radius), dim=1)
    cell_feature = torch.cat((cell_feature, cell_volume), dim=1)
    if torch.any(torch.isnan(cell_feature)):
        cell_feature[torch.isnan(cell_feature)] = 0.
    return cell_feature, cell_ball_center


def get_facet_feature(facet_vertex_idx, current_xyz, facet_nei_cell, infinite_cell, cell_ball_centers):
    facet_vertex_idx = facet_vertex_idx.long()
    facet_vertex_a = current_xyz[facet_vertex_idx[:, 0]]
    facet_vertex_b = current_xyz[facet_vertex_idx[:, 1]]
    facet_vertex_c = current_xyz[facet_vertex_idx[:, 2]]
    ab = facet_vertex_b - facet_vertex_a
    bc = facet_vertex_c - facet_vertex_b
    ca = facet_vertex_a - facet_vertex_c
    fn = f.normalize(torch.cross(ab, -1.0 * ca, dim=1), p=2, dim=-1)
    facet_area = torch.norm(torch.cross(ab, -ca, dim=1), dim=1, keepdim=True) / 2.0  # (t,1)
    side_length = torch.cat((torch.norm(ab, dim=1, keepdim=True), torch.norm(bc, dim=1, keepdim=True)), dim=1)
    side_length = torch.cat((side_length, torch.norm(ca, dim=1, keepdim=True)), dim=1)
    length_max = torch.max(side_length, dim=1, keepdim=True)[0]
    length_min = torch.min(side_length, dim=1, keepdim=True)[0]
    length = torch.cat((length_max, length_min), dim=1)
    angle = []
    for i in range(2):
        cell_index = facet_nei_cell[:, i]
        oa = facet_vertex_a - cell_ball_centers[cell_index.long()]
        cos = torch.sum(torch.mul(fn, oa), dim=1, keepdim=True)
        cos = cos / (torch.norm(fn, dim=1, keepdim=True) * torch.norm(oa, dim=1, keepdim=True))
        cos = torch.abs(cos)
        angle.append(cos)
    angle = torch.cat((angle[0], angle[1]), dim=1)
    for i in range(infinite_cell.shape[0]):
        angle[facet_nei_cell == infinite_cell[i]] = 1.0
    angle = torch.min(angle, dim=1, keepdim=True)[0]
    angle = 1.0 - angle

    facet_feature = torch.cat((length, facet_area), dim=1)
    facet_feature = torch.cat((facet_feature, angle), dim=1)

    if torch.any(torch.isnan(facet_feature)):
        facet_feature[torch.isnan(facet_feature)] = 0.
    return facet_feature, length_max


def round_computation(point1, point2, point3):
    x1 = point1[:, 0].reshape(-1, 1)
    y1 = point1[:, 1].reshape(-1, 1)
    z1 = point1[:, 2].reshape(-1, 1)
    x2 = point2[:, 0].reshape(-1, 1)
    y2 = point2[:, 1].reshape(-1, 1)
    z2 = point2[:, 2].reshape(-1, 1)
    x3 = point3[:, 0].reshape(-1, 1)
    y3 = point3[:, 1].reshape(-1, 1)
    z3 = point3[:, 2].reshape(-1, 1)

    a1 = (y1 * z2 - y2 * z1 - y1 * z3 + y3 * z1 + y2 * z3 - y3 * z2)
    b1 = -(x1 * z2 - x2 * z1 - x1 * z3 + x3 * z1 + x2 * z3 - x3 * z2)
    c1 = (x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2)
    d1 = -(x1 * y2 * z3 - x1 * y3 * z2 - x2 * y1 * z3 + x2 * y3 * z1 + x3 * y1 * z2 - x3 * y2 * z1)

    a2 = 2 * (x2 - x1)
    b2 = 2 * (y2 - y1)
    c2 = 2 * (z2 - z1)
    d2 = x1 * x1 + y1 * y1 + z1 * z1 - x2 * x2 - y2 * y2 - z2 * z2

    a3 = 2 * (x3 - x1)
    b3 = 2 * (y3 - y1)
    c3 = 2 * (z3 - z1)
    d3 = x1 * x1 + y1 * y1 + z1 * z1 - x3 * x3 - y3 * y3 - z3 * z3

    x = -(b1 * c2 * d3 - b1 * c3 * d2 - b2 * c1 * d3 + b2 * c3 * d1 + b3 * c1 * d2 - b3 * c2 * d1) / (
            a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)
    y = (a1 * c2 * d3 - a1 * c3 * d2 - a2 * c1 * d3 + a2 * c3 * d1 + a3 * c1 * d2 - a3 * c2 * d1) / (
            a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)
    z = -(a1 * b2 * d3 - a1 * b3 * d2 - a2 * b1 * d3 + a2 * b3 * d1 + a3 * b1 * d2 - a3 * b2 * d1) / (
            a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)
    r = torch.sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y) + (z1 - z) * (z1 - z))
    circle_center = torch.cat((x, y), dim=1)
    circle_center = torch.cat((circle_center, z), dim=1)
    return circle_center, r
