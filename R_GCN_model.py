import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import losses

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

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
        layer_normalize=False,
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
        self.layer_normalize = layer_normalize
        self.activation_fn = activation_fn

    def forward(self, input):

        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.layer_normalize:
            layer_norm = nn.LayerNorm(x.size()[1:], eps=1e-6)
            x = layer_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


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

    def __init__(self, d_in, d_middle, d_out, activation_fn=nn.LeakyReLU(0.02)):
        super(res_block, self).__init__()

        self.conv2d_1 = SharedMLP(d_in, d_middle, bn=True, activation_fn=activation_fn)
        self.conv2d_2 = SharedMLP(d_middle, d_out, bn=True)
        self.shortcut = SharedMLP(d_in, d_out, bn=True)
        self.lrelu = activation_fn

    def forward(self, batch_feature):

        output = self.conv2d_1(batch_feature)
        output = self.conv2d_2(output)
        res = self.shortcut(batch_feature)
        output = output + res
        output = self.lrelu(output)

        return output



class R_GCN(nn.Module):
    def __init__(self, geo_in):
        super(R_GCN, self).__init__()

        self.point_encoder = nn.Sequential(
            res_block(geo_in, 32, 64),
            res_block(64, 128, 64),
        )
        self.attention_p = att_pooling(64,16,activation_fn=nn.LeakyReLU(0.02))

        self.dropout = nn.Dropout(0.2)
        self.conv2d_1 = SharedMLP(64, 16, bn=True, activation_fn=nn.LeakyReLU(0.02))

        self.conv2d_21 = SharedMLP(128, 16, bn=True, activation_fn=nn.ReLU())
        self.conv2d_22 = SharedMLP(128, 16, bn=True, activation_fn=nn.ReLU())
        self.conv2d_23 = SharedMLP(128, 16, bn=True, activation_fn=nn.ReLU())

        self.node_encoder = nn.Sequential(
            res_block(16+3, 64, 16),
        )
        self.attention_0 = att_pooling(16, 16, activation_fn=nn.LeakyReLU(0.02))
        self.attention_1 = att_pooling(16, 16, activation_fn=nn.LeakyReLU(0.02))
        self.cycle_facet_encoder_1 = nn.Sequential(
            res_block(20+20, 64, 128, activation_fn=nn.ReLU()),
        )
        self.cycle_node_encoder_1 = nn.Sequential(
            res_block(20+20+16, 64, 128, activation_fn=nn.ReLU()),
        )
        self.cycle_facet_encoder_2 = nn.Sequential(
            res_block(16+16, 64, 128, activation_fn=nn.ReLU()),
        )
        self.cycle_node_encoder_2 = nn.Sequential(
            res_block(16+16+16, 64, 128, activation_fn=nn.LeakyReLU(0.01)),
            res_block(128, 256, 128, activation_fn=nn.LeakyReLU(0.01)),
        )
        self.conv2d_3 = SharedMLP(128, 2)
        self.attention_2 = att_pooling(128, 2, last_layer=True)

    def forward(self, deepdt_data):
        point_feature = deepdt_data.batch_point_feature.permute(2, 0, 1).unsqueeze(0)
        point_feature = self.point_encoder(point_feature)
        point_feature = point_feature.squeeze(0).permute(1,2,0)
        point_feature = self.attention_p(point_feature)
        node_feature = feature_fetch(point_feature, deepdt_data.cell_vertex_idx)
        node_feature = torch.cat((node_feature, deepdt_data.cell_offset_xyz), dim=2)
        node_feature = self.node_encoder(node_feature.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        node_feature = self.attention_0(node_feature)
        node_feature = torch.cat((node_feature, deepdt_data.batch_cell_feature), dim=1)
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
        node_neighbor_feature = node_feature[deepdt_data.adj_idx.long()]
        facet_neighbor_feature = feature_fetch(facet_feature, deepdt_data.tet_adj_facet)
        last_dim = node_feature.shape[1]
        node_feature = node_feature.reshape(-1, 1, last_dim).expand(-1, 4, last_dim)
        node_feature = torch.cat((node_feature, node_neighbor_feature), dim=2)
        node_feature = torch.cat((node_feature, facet_neighbor_feature), dim=2)
        node_feature = node_feature.permute(2, 0, 1).unsqueeze(0)
        node_feature = self.cycle_node_encoder_2(node_feature).squeeze(0).permute(1, 2, 0)
        cell_pred = self.attention_2(node_feature)
        cell_pred_soft = F.softmax(cell_pred, dim=1)

        loss1, loss2, loss3 = losses.build_losses(cell_pred, cell_pred_soft, deepdt_data)

        return cell_pred_soft, loss1, loss2, loss3








