import os
import numpy as np
import torch
from torch.utils.data import Dataset

import DT_data

class DTUDelPara:
    def __init__(self):
        self.class_freq = np.asarray([50.0, 50.0])
        self.class_weights = np.asarray([1.0, 1.0], dtype=np.float32)
        self.num_classes = 2
        self.color_map = [[255, 0, 0],  # 0 (red)
                          [0, 255, 0]]  # 1 (green)

class DTUDelDataset(Dataset):
    def __init__(self, cfg, mode, transform=None):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.scan_paths, self.scan_names = self.load_scan_paths()

    def load_scan_paths(self):
        if self.mode == "train":
            data_name = self.cfg["train_name_file"]
        elif self.mode == "test":
            data_name = self.cfg["test_name_file"]
        elif self.mode == "val":
            data_name = self.cfg["val_name_file"]
        else:
            data_name = self.cfg["test_name_file"]

        with open(data_name) as f:
            scan_names = f.readlines()

        scan_names = [s.rstrip() for s in scan_names]
        scan_names = [s for s in scan_names if s]
        if self.mode == "val":
            return [os.path.join(self.cfg["val_data_root_dir"], s.rstrip()) for s in scan_names], scan_names
        else:
            return [os.path.join(self.cfg["data_root_dir"], s.rstrip()) for s in scan_names], scan_names

    def __len__(self):
        return len(self.scan_names)

    def load_scan(self, idx):
        s = DT_data.ScanData()
        s.data_para = DTUDelPara()
        s.scan_name = self.scan_names[idx]
        s.load_full_scan(self.scan_paths[idx], self.cfg)

        return s

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        scan = self.load_scan(idx)
        sample = DT_data.create_full_data(scan, self.cfg)

        return sample





