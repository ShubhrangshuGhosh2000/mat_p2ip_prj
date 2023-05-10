from torch.utils.data import Dataset

# create custom dataset class
class CustomDataset(Dataset):
    # constuctor
    def __init__(self, feat_arr_2d, label_arr_1d):
        super(CustomDataset, self).__init__()
        self.feat_arr_2d = feat_arr_2d
        self.label_arr_1d = label_arr_1d

    # return the size of the dataset
    def __len__(self):
        return len(self.label_arr_1d)

    # fetch a data sample for a given index
    def __getitem__(self, idx):
        feat_arr_1d = self.feat_arr_2d[idx, :]
        label = self.label_arr_1d[idx]
        sample = (feat_arr_1d, label)
        return sample
