import os
import numpy as np
from torch.utils.data import Dataset

def read_ti(fn):
    data = []
    with open(fn) as f:
        for line in f:
            line = line.strip().split()
            data.append(line)
    data = np.array(data, dtype="float64")
    data = np.delete(data, 0, axis=0)  # drop header
    return data

class pointdata(Dataset):
    def __init__(self, input_path, output_path, data_info, point_number):
        self._input_path = input_path
        self.file_ids = sorted(
            [int(f[:-4]) for f in os.listdir(input_path) if f.endswith(".txt")]
        )
        self.point_number = point_number

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        fileNo = self.file_ids[index]
        data = read_ti(os.path.join(self._input_path, f"{fileNo}.txt"))
        np.random.shuffle(data)

        xyz_raw = data[:, 1:4].copy()
        label_raw = data[:, 4:8].copy()  # u v w p

        eps = 1e-12
        lmin = label_raw.min(axis=0)
        lmax = label_raw.max(axis=0)
        xmin = xyz_raw.min(axis=0)
        xmax = xyz_raw.max(axis=0)


        label = (label_raw - lmin) / (lmax - lmin + eps)
        data[:, 1:4] = (xyz_raw - xmin) / (xmax - xmin + eps)
        x_in = data[:, 0:4]  
        return x_in.T.astype(np.float32), \
               label.T.astype(np.float32), \
               lmin.astype(np.float32), \
               lmax.astype(np.float32)
