from dataloader import load_dataset
import glob

class Dataset:
    def __init__(self, data_path, drop_rate=0.3, seed=None):
        self.file_list = glob.glob(data_path + "/*.json")
        self.samples = []  # List of (partial, full) graph pairs
        self.drop_rate = drop_rate
        self.seed = seed

        for file_path in self.file_list:
            full_list, partial_list = load_dataset([file_path], drop_rate=self.drop_rate, seed=self.seed)
            for partial, full in zip(partial_list, full_list):
                self.samples.append((partial, full))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
