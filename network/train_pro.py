import argparse
import os

import train_utils
import DTU
from DT_data import *
from DT_dataloader import *
import R_GCN_model
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="./train_cfg.yaml", type=str, help='Path to config file')

args = parser.parse_args()
cfg = train_utils.load_config(args.config)
cfg = train_utils.augment_config(cfg)
cfg = train_utils.check_config(cfg)

print(cfg)

if not os.path.exists(cfg["experiment_dir"]):
    os.makedirs(cfg["experiment_dir"])

geo_in = 6
train_model = R_GCN_model.R_GCN(geo_in)
model_path = cfg["model_path"]

if cfg["cuda"]:
    train_model = DTParallel(train_model, device_ids=cfg["device_ids"])
    device = torch.device("cuda:{}".format(cfg["device_ids"][0]))
    train_model = train_model.to(device)

if cfg["pretrained"]:
    if os.path.exists(model_path):
        train_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("pretrained model loaded")
    else:
        print("training model from scratch")
else:
    print("training model from scratch")

optimizer = torch.optim.Adam(train_model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, [5, 10, 15], gamma=0.5)
train_data = DTU.DTUDelDataset(cfg, "train")
train_data_loader = DataListLoader(train_data, cfg["batch_size"], shuffle=True,num_workers=cfg["num_workers"])
val_data = DTU.DTUDelDataset(cfg, "val")
val_data_loader = DataListLoader(val_data, cfg["batch_size"], num_workers=cfg["num_workers"])

step_cnt = 0
best_accu = 0.0
best_loss = 0.0
output_dir = os.path.split(model_path)[0]

init_epoch = 0
epoch_path = os.path.join(output_dir, "epoch.txt")
if os.path.exists(epoch_path):
    with open(os.path.join(output_dir, "epoch.txt"), 'r') as f:
        init_epoch = int(f.read())
    print("init_epoch loaded", init_epoch)
print("init_epoch", init_epoch)

weight1 = 1.0 * cfg["weight_ratio"][0]
weight2 = 1.0 * cfg["weight_ratio"][1]
weight3 = 1.0 * cfg["weight_ratio"][2]
print("loss1 weight", weight1, 'loss2 weight', weight2, 'loss3 weight', weight3)

tmp_cnt = 0
tmp_label0 = 0
tmp_label1 = 0
tmp_loss1 = 0.0
tmp_loss2 = 0.0
tmp_loss3 = 0.0
tmp_loss = 0.0
loss1_best = 100.0
# tensorboard
writer = SummaryWriter(cfg["SummaryWriter_path"])
data_lists = train_data_loader
val_data_lists = val_data_loader

for epoch in range(init_epoch, cfg['epochs']):
    for data_list in data_lists:
        for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        train_model.train()
        optimizer.zero_grad()

        cell_pred, loss1, loss2, loss3 = train_model(data_list)
        loss = weight1 * loss1 + weight2 * loss2 + weight3 * loss3
        loss.backward()
        optimizer.step()

        step_cnt += 1
        cell_pred_label = cell_pred.max(dim=1)[1]
        label0_num = torch.sum(cell_pred_label == 0).item()
        label1_num = torch.sum(cell_pred_label == 1).item()
        outstr = "Train epoch %d, step %d, loss %.6f, loss1 %.6f, loss2 %.6f, loss3 %.6f" \
                 % (epoch, step_cnt, loss.detach().item(), loss1.detach().item(), loss2.detach().item(),
                    loss3.detach().item())
        print(outstr)
        for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()

        tmp_loss1 += loss1.detach().item()
        tmp_loss2 += loss2.detach().item()
        tmp_loss3 += loss3.detach().item()
        tmp_loss += loss.detach().item()
        tmp_cnt += 1
        tmp_label0 += label0_num
        tmp_label1 += label1_num

        if tmp_cnt % 500 == 0 and tmp_cnt != 0:
            extra_output_dir = os.path.split(model_path)[0] + '_epoch_' + str(epoch)
            if not os.path.exists(extra_output_dir):
                os.mkdir(extra_output_dir)
            extra_model_path = os.path.join(extra_output_dir, os.path.split(model_path)[1])
            print("Saving extra model", cfg["weight_ratio"])
            torch.save(train_model.state_dict(), extra_model_path)

    tmp_loss1 /= tmp_cnt
    tmp_loss2 /= tmp_cnt
    tmp_loss3 /= tmp_cnt
    tmp_loss /= tmp_cnt

    exp_lr_scheduler.step()

    outstr = "Test epoch %d, step %d, loss %.6f, loss1 %.6f, loss2 %.6f, loss3 %.6f" \
             % (epoch, step_cnt, tmp_loss, tmp_loss1, tmp_loss2, tmp_loss3)
    print(outstr)

    output_dir = os.path.split(model_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model", cfg["weight_ratio"])

    if epoch % 1 == 0:
        val_tmp_loss, val_tmp_loss1, val_tmp_loss2, val_tmp_loss3 = \
            train_utils.val(val_data_lists, cfg, train_model, weight1, weight2, weight3)
        writer.add_scalar("val_tmp_loss", val_tmp_loss, epoch)
        writer.add_scalar("val_tmp_loss1", val_tmp_loss1, epoch)
        writer.add_scalar("val_tmp_loss2", val_tmp_loss2, epoch)
        writer.add_scalar("val_tmp_loss3", val_tmp_loss3, epoch)
        if val_tmp_loss1 <= loss1_best:
            loss1_best = val_tmp_loss1
            torch.save(train_model.state_dict(), model_path)
            loss_array = np.asarray([val_tmp_loss, val_tmp_loss1, val_tmp_loss2, val_tmp_loss3])
            np.savetxt(os.path.join(output_dir, 'loss.txt'), loss_array, fmt='%f')
            with open(os.path.join(output_dir, "epoch.txt"), 'w') as f:
                f.write(str(epoch))

        extra_output_dir = os.path.split(model_path)[0] + '_epoch_' + str(epoch)
        if not os.path.exists(extra_output_dir):
            os.makedirs(extra_output_dir)
        extra_model_path = os.path.join(extra_output_dir, os.path.split(model_path)[1])
        print("Saving extra model", cfg["weight_ratio"])
        torch.save(train_model.state_dict(), extra_model_path)
        loss_array = np.asarray([val_tmp_loss, val_tmp_loss1, val_tmp_loss2, val_tmp_loss3])
        np.savetxt(os.path.join(extra_output_dir, 'loss.txt'), loss_array, fmt='%f')
        with open(os.path.join(extra_output_dir, "epoch.txt"), 'w') as f:
            f.write(str(epoch))

    writer.add_scalar("tmp_loss", tmp_loss, epoch)
    writer.add_scalar("tmp_loss1", tmp_loss1, epoch)
    writer.add_scalar("tmp_loss2", tmp_loss2, epoch)
    writer.add_scalar("tmp_loss3", tmp_loss3, epoch)

    tmp_cnt = 0
    tmp_label0 = 0
    tmp_label1 = 0
    tmp_loss1 = 0.0
    tmp_loss2 = 0.0
    tmp_loss3 = 0.0
    tmp_loss = 0.0
