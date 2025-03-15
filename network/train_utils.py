import torch
import os
import numpy as np
import yaml


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def augment_config(cfg):
    cfg["model_path"] = os.path.join(cfg["experiment_dir"], cfg["model_path"])
    cfg["SummaryWriter_path"] = os.path.join(cfg["experiment_dir"], cfg["SummaryWriter_path"])

    return cfg


def check_config(cfg):
    if cfg["batch_size"] > len(cfg["device_ids"]):
        print("Warning: batch_size must be no more than device count")
        print("Set new batch_size as", len(cfg["device_ids"]))
        cfg["batch_size"] = len(cfg["device_ids"])

    return cfg


def val(val_data_lists, cfg, train_model, weight1, weight2, weight3):
    val_tmp_loss1 = 0.0
    val_tmp_loss2 = 0.0
    val_tmp_loss3 = 0.0
    val_tmp_loss = 0.0
    val_num = 0
    for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
        with torch.cuda.device(d):
            torch.cuda.empty_cache()
    for val_data_list in val_data_lists:
        train_model.eval()
        with torch.no_grad():
            val_cell_pred, val_loss1, val_loss2, val_loss3 = train_model(val_data_list)
            val_loss = weight1 * val_loss1 + weight2 * val_loss2 + weight3 * val_loss3
            val_tmp_loss1 += val_loss1.detach().item()
            val_tmp_loss2 += val_loss2.detach().item()
            val_tmp_loss3 += val_loss3.detach().item()
            val_tmp_loss += val_loss.detach().item()
            val_num += 1
    val_tmp_loss1 /= val_num
    val_tmp_loss2 /= val_num
    val_tmp_loss3 /= val_num
    val_tmp_loss /= val_num
    print("val_tmp_loss1", val_tmp_loss1, "-val_tmp_loss2", val_tmp_loss2, "-val_tmp_loss3", val_tmp_loss3, "-val_tmp_loss", val_tmp_loss)

    return val_tmp_loss, val_tmp_loss1, val_tmp_loss2, val_tmp_loss3


def relabeling(labels, infinite_cell_id, cell_adj_id):
    labels = labels.unsqueeze(-1)
    adj_labels = labels[cell_adj_id.long()].squeeze(-1)
    adj_labels_sum = torch.sum(adj_labels, dim=-1, keepdim=True)
    inside = torch.where(adj_labels_sum == 5)
    outside = torch.where(adj_labels_sum == 7)
    labels[inside] = int(1)
    labels[outside] = int(2)
    labels[infinite_cell_id.long()] = int(2)

    return labels.squeeze(-1)
