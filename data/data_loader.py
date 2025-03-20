import pickle

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, data_path, device):
        self.data_dir = data_path
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        # return size of dataset
        return len(self.data)

    def __getitem__(self, idx):
        a, g, H, Z = self.data[idx]
        # H_r, H_i, Z_r, Z_i= H.real, H.imag, Z.real, Z.imag
        # return S, a, G, Sigma_hat
        return a.to(self.device), g.to(self.device), H.to(self.device), Z.to(self.device)


def make_data_loader(data_path, batch_size, device, shuffle):
    datasets = MyDataset(data_path, device)
    data_loader = DataLoader(
        datasets, batch_size, shuffle
    )

    return data_loader
