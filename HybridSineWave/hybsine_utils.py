import numpy as np
from torch.utils.data import Dataset


class HybridSineSet(Dataset):
    def __init__(self, dataset):
        self.data = []
        self.tps = []
        self.length = []

        max_len = max([len(d[4]) for d in dataset])

        fill_tps = np.linspace(9990, 9999, max_len)
        for i, (_, _, _, d, tp) in enumerate(dataset):
            seg_len = len(tp)

            zero_array = np.zeros((max_len - seg_len))
            tp_zero_array = fill_tps[-(max_len-seg_len):]

            d = np.concatenate((d, zero_array))
            d = np.expand_dims(d, -1)
            self.data.append(d)

            self.tps.append(np.concatenate((tp, zero_array)))
            self.length.append(seg_len)

    def __len__(self):
        return len(self.length)

    def __getitem__(self, idx):
        return self.data[idx], self.tps[idx], self.length[idx]


class SegmentedSineSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        data = []
        tps = []

        max_len = 0

        for i in range(len(dataset)):
            traj_len = len(dataset[i][1])

            cps = list(dataset[i][2])
            cps = [0] + cps + [traj_len]

            for j in range(len(cps) - 1):
                d = dataset[i][0][cps[j]:cps[j + 1]]
                t = dataset[i][1][cps[j]:cps[j + 1]]

                max_len = max(max_len, len(d))
                data.append(d)
                tps.append(t - t[0])

        length = []

        fill_tps = np.linspace(9990, 9999, max_len)
        for i in range(len(data)):
            seg_len = len(data[i])
            length.append(seg_len)

            zero_array = np.zeros((max_len - seg_len))
            d = np.concatenate((data[i], zero_array))
            d = np.expand_dims(d, -1)
            data[i] = d

            tp_zero_array = fill_tps[-(max_len - seg_len):]
            tps[i] = np.concatenate((tps[i], zero_array))

        self.data = np.array(data)
        self.tps = np.array(tps)
        self.length = length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.tps[idx], self.length[idx]
