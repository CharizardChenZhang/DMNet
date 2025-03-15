import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import losses

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def random_sample(feature, sample_idx):

    pool_features = feature[sample_idx]
    pool_features = torch.max(pool_features, 1, False)[0]
    return pool_features


def gather_neighbour(feature, neighbor_idx):


    neighbor_features = feature[neighbor_idx]
    return neighbor_features


def feature_fetch(input_tensor, index_tensor):

    feature = input_tensor[index_tensor.long()]
    feature[index_tensor == -1] = 0

    return feature


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode=0,
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):

        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class multi_att_pooling(nn.Module):

    def __init__(self, d_in, d_out, dropout=0.5, num_heads=8, activation_fn=nn.ReLU(), last_layer=False):
        super(multi_att_pooling, self).__init__()

        self.num_heads = num_heads
        head_dim = d_in // num_heads
        self.scale = head_dim ** (-0.5)
        self.qkv = nn.Linear(d_in, 3*d_in, bias=False)
        if last_layer:
            self.mlp = SharedMLP(d_in, d_out)
        else:
            self.mlp = SharedMLP(d_in, d_out, bn=True, activation_fn=activation_fn)
        self.dropout = dropout


    def forward(self, feature_set):

        B, N, C = feature_set.shape
        qkv = self.qkv(feature_set).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention_scores = torch.matmul(q, k.transpose(-2, -1))*self.scale
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = F.dropout(attention_probs, self.dropout, training=self.training)

        f_agg = torch.matmul(attention_probs, v).transpose(1,2).reshape(B, N, C)
        f_agg = torch.max(f_agg, dim=1, keepdim=True)[0]
        f_agg = f_agg.permute(2,0,1).unsqueeze(0)
        f_agg = self.mlp(f_agg)
        f_agg = f_agg.squeeze(3).squeeze(0).permute(1,0)

        return f_agg


class att_pooling(nn.Module):

    def __init__(self, d_in, d_out, activation_fn=nn.ReLU(), last_layer=False):
        super(att_pooling, self).__init__()

        self.fc = nn.Linear(d_in, d_in, bias=False)
        if last_layer:
            self.mlp = SharedMLP(d_in, d_out)
        else:
            self.mlp = SharedMLP(d_in, d_out, bn=True, activation_fn=activation_fn)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = torch.softmax(att_activation, 1)
        f_agg = feature_set.mul(att_scores)
        f_agg = torch.sum(f_agg, dim=1, keepdim=True)
        f_agg = f_agg.permute(2,0,1).unsqueeze(0)
        f_agg = self.mlp(f_agg)
        f_agg = f_agg.squeeze(3).squeeze(0).permute(1,0)

        return f_agg


class res_block(nn.Module):

    def __init__(self, d_in, d_middle, d_out, activation_fn=nn.LeakyReLU(0.2)):
        super(res_block, self).__init__()

        self.conv2d_1 = SharedMLP(d_in, d_middle, bn=True, activation_fn=activation_fn)
        self.conv2d_2 = SharedMLP(d_middle, d_out, bn=True)
        self.shortcut = SharedMLP(d_in, d_out)
        self.lrelu = activation_fn

    def forward(self, batch_feature):

        output = self.conv2d_1(batch_feature)
        output = self.conv2d_2(output)
        res = self.shortcut(batch_feature)
        output = output + res
        output = self.lrelu(output)

        return output


class GraphConvolution(Module):
    """
    Simple GCN, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class R_GCN(nn.Module):
    def __init__(self, geo_in):
        super(R_GCN, self).__init__()

        self.point_encoder = nn.Sequential(
            res_block(geo_in, 32, 64),
            res_block(64, 128, 256),
            res_block(256, 128, 64),
        )
        self.attention_p = att_pooling(64,16,activation_fn=nn.LeakyReLU(0.2))

        self.dropout = nn.Dropout(0.2)
        self.conv2d_1 = SharedMLP(64, 16, bn=True, activation_fn=nn.LeakyReLU(0.2))

        self.conv2d_21 = SharedMLP(128, 16, bn=True, activation_fn=nn.ReLU())
        self.conv2d_22 = SharedMLP(128, 16, bn=True, activation_fn=nn.ReLU())
        self.conv2d_23 = SharedMLP(128, 16, bn=True, activation_fn=nn.ReLU())

        self.node_encoder = nn.Sequential(
            res_block(16+3, 64, 16),
        )
        self.attention_0 = att_pooling(16, 16, activation_fn=nn.LeakyReLU(0.2))
        self.attention_1 = att_pooling(16, 16, activation_fn=nn.LeakyReLU(0.2))

        # cycle_facet_1
        self.cycle_facet_encoder_1 = nn.Sequential(
            res_block(20+20, 64, 128, activation_fn=nn.ReLU()),
            res_block(128, 256, 128, activation_fn=nn.ReLU()),
        )
        # cycle_node_1
        self.cycle_node_encoder_1 = nn.Sequential(
            res_block(20+20+16, 64, 128, activation_fn=nn.ReLU()),
            res_block(128, 256, 128, activation_fn=nn.ReLU()),
        )
        # cycle_facet_2
        self.cycle_facet_encoder_2 = nn.Sequential(
            res_block(16+16, 64, 128, activation_fn=nn.ReLU()),
            res_block(128, 256, 128, activation_fn=nn.ReLU()),
        )
        # cycle_node_2
        self.cycle_node_encoder_2 = nn.Sequential(
            res_block(16+16+16, 64, 128, activation_fn=nn.ReLU()),
            res_block(128, 256, 128, activation_fn=nn.ReLU()),
        )
        self.conv2d_3 = SharedMLP(128, 2)
        self.attention_2 = multi_att_pooling(128, 2, dropout=0.5, last_layer=True)

    def forward(self, deepdt_data):
        point_feature = deepdt_data.batch_point_feature.permute(2, 0, 1).unsqueeze(0)
        point_feature = self.point_encoder(point_feature)
        point_feature = point_feature.squeeze(0).permute(1,2,0)
        point_feature = self.attention_p(point_feature)

        # encoding initial node_feature
        node_feature = feature_fetch(point_feature, deepdt_data.cell_vertex_idx)
        node_feature = torch.cat((node_feature, deepdt_data.cell_offset_xyz), dim=2)
        node_feature = self.node_encoder(node_feature.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        node_feature = self.attention_0(node_feature)  # (m,32)
        node_feature = torch.cat((node_feature, deepdt_data.batch_cell_feature), dim=1)

        # encoding initial facet_feature
        facet_feature = feature_fetch(point_feature, deepdt_data.facet_vertex_idx)
        facet_feature = self.attention_1(facet_feature)
        facet_feature = torch.cat((facet_feature, deepdt_data.batch_facet_feature), dim=1)

        node_neighbor_feature = node_feature[deepdt_data.facet_nei_cell.long()]
        last_dim = facet_feature.shape[1]
        facet_feature = facet_feature.reshape(-1, 1, last_dim).expand(-1, 2, last_dim)
        facet_feature = torch.cat((facet_feature, node_neighbor_feature), dim=2)
        facet_feature = facet_feature.permute(2, 0, 1).unsqueeze(0)
        facet_feature = self.cycle_facet_encoder_1(facet_feature)
        facet_feature = torch.max(facet_feature, dim=3, keepdim=True)[0]
        facet_feature = self.conv2d_21(self.dropout(facet_feature)).squeeze(3).squeeze(0).permute(1, 0)

        node_neighbor_feature = node_feature[deepdt_data.adj_idx.long()]
        facet_neighbor_feature = feature_fetch(facet_feature, deepdt_data.tet_adj_facet)
        last_dim = node_feature.shape[1]
        node_feature = node_feature.reshape(-1, 1, last_dim).expand(-1, 4, last_dim)
        node_feature = torch.cat((node_feature, node_neighbor_feature), dim=2)
        node_feature = torch.cat((node_feature, facet_neighbor_feature), dim=2)
        node_feature = node_feature.permute(2, 0, 1).unsqueeze(0)
        node_feature = self.cycle_node_encoder_1(node_feature)
        node_feature = torch.max(node_feature, dim=3, keepdim=True)[0]
        node_feature = self.conv2d_22(self.dropout(node_feature)).squeeze(3).squeeze(0).permute(1, 0)

        node_neighbor_feature = node_feature[deepdt_data.facet_nei_cell.long()]
        last_dim = facet_feature.shape[1]
        facet_feature = facet_feature.reshape(-1, 1, last_dim).expand(-1, 2, last_dim)
        facet_feature = torch.cat((facet_feature, node_neighbor_feature), dim=2)
        facet_feature = facet_feature.permute(2, 0, 1).unsqueeze(0)
        facet_feature = self.cycle_facet_encoder_2(facet_feature)
        facet_feature = torch.max(facet_feature, dim=3, keepdim=True)[0]
        facet_feature = self.conv2d_23(self.dropout(facet_feature)).squeeze(3).squeeze(0).permute(1, 0)

        node_neighbor_feature = node_feature[deepdt_data.adj_idx.long()]
        facet_neighbor_feature = feature_fetch(facet_feature, deepdt_data.tet_adj_facet)
        last_dim = node_feature.shape[1]
        node_feature = node_feature.reshape(-1, 1, last_dim).expand(-1, 4, last_dim)
        node_feature = torch.cat((node_feature, node_neighbor_feature), dim=2)
        node_feature = torch.cat((node_feature, facet_neighbor_feature), dim=2)
        node_feature = node_feature.permute(2, 0, 1).unsqueeze(0)
        node_feature = self.cycle_node_encoder_2(node_feature).squeeze(0).permute(1, 2, 0)
        # node_feature = torch.max(node_feature, dim=3, keepdim=True)[0]
        # cell_pred = self.conv2d_3(node_feature).squeeze(3).squeeze(0).permute(1,0)
        cell_pred = self.attention_2(node_feature)
        cell_pred_soft = F.softmax(cell_pred, dim=1)

        loss1, loss2, loss3 = losses.build_losses(cell_pred, cell_pred_soft, deepdt_data)

        return cell_pred_soft, loss1, loss2, loss3







