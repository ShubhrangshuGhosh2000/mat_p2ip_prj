from torch.utils.data import Dataset

# create custom dataset class
class Custom2DfeatDataset(Dataset):
    # constuctor
    def __init__(self, two_d_feat_arr_1, two_d_feat_arr_2, label_arr_1d):
        super(Custom2DfeatDataset, self).__init__()
        self.two_d_feat_arr_1 = two_d_feat_arr_1
        self.two_d_feat_arr_2 = two_d_feat_arr_2
        self.label_arr_1d = label_arr_1d

    # return the size of the dataset
    def __len__(self):
        return len(self.two_d_feat_arr_1)

    # fetch a data sample for a given index
    def __getitem__(self, idx):
        two_d_feat_1 = self.two_d_feat_arr_1[idx]
        two_d_feat_2 = self.two_d_feat_arr_2[idx]
        label = self.label_arr_1d[idx]
        sample = (two_d_feat_1, two_d_feat_2, label)
        return sample
