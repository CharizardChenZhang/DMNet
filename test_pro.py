import argparse
import train_utils
import DTU
from DMNet_data import *
from Parallel import *
import R_GCN_model
import dataloader
import numpy as np
from time import *
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file')
args = parser.parse_args()
cfg = train_utils.load_config(args.config)
cfg = train_utils.augment_config(cfg)
cfg = train_utils.check_config(cfg)

geo_in = 6
train_model = R_GCN_model.R_GCN(geo_in)
model_path = cfg["model_path"]

if cfg["cuda"]:
    train_model = DMNetParallel(train_model, device_ids=cfg["device_ids"])
    device = torch.device("cuda:{}".format(cfg["device_ids"][0]))
    train_model = train_model.to(device)

if cfg["pretrained"]:
    if os.path.exists(model_path):
        d = torch.load(model_path, map_location="cpu")
        train_model.load_state_dict(d)
        print("pretrained model loaded")
    else:
        print("training model from scratch")
else:
    print("training model from scratch")

test_data = DTU.DTUDelDataset(cfg, "test")
test_data_loader = dataloader.DataListLoader(test_data, cfg["batch_size"], num_workers=cfg["num_workers"])

for test_data_list in test_data_loader:
    for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
        with torch.cuda.device(d):
            torch.cuda.empty_cache()

    train_model.eval()
    with torch.no_grad():
        time1 = time()
        cell_pred, loss1, loss2, loss3 = train_model(test_data_list)
        preds = cell_pred.max(dim=1)[1]
        time2 = time()
        print("consume time", time2-time1,"s")

        labels_pr = preds.detach().cpu() + 1
        cnt = 0
        for data in test_data_list:
            label_num = data.cell_vertex_idx.shape[0]
            label_begin = cnt
            label_end = cnt + label_num
            cnt += label_num
            data_labels_pr = labels_pr[label_begin:label_end]
            data_labels_pr[data.infinite_cell.long()] = 2.
            data_labels_pr = data_labels_pr.numpy()

            loss1_pr = loss1.item()
            loss2_pr = loss2.item()
            loss3_pr = loss3.item()
            output_dir = cfg["experiment_dir"]
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_dir = os.path.join(cfg["experiment_dir"], data.data_name.split("/")[0])
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_dir = os.path.join(cfg["experiment_dir"], data.data_name)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            np.savetxt(os.path.join(output_dir, "pre_label.txt"), data_labels_pr, fmt='%d')
            print('loss1 %.6f, loss2 %.6f, loss3 %.6f' % (loss1_pr, loss2_pr, loss3_pr))
            print("test", data.data_name, "done.")

    for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
        with torch.cuda.device(d):
            torch.cuda.empty_cache()



















