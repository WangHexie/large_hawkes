import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class NewsDataset:
    def __init__(self):
        self._read()
        self._process()

    def _read(self):
        self.file_data = pd.read_json("../../datasets/data_d_a_f.json", lines=True)

    def _process(self):
        self.file_data.drop_duplicates(subset="content", inplace=True)
        self.file_data = self.file_data[self.file_data['publish_author'] != '']
        self.file_data = self.file_data.sort_values(by='publish_time')

    def get_data(self, max_length=10000, number_of_publication=5):
        count_data = self.file_data[-max_length:].groupby("publish_author").count()
        selected_data = self.file_data[-max_length:][
            self.file_data["publish_author"].isin(count_data.index[count_data["title"] > number_of_publication])]
        all_writer = list(set(selected_data["publish_author"].values))
        id_map = {all_writer[i]: i for i in range(len(all_writer))}
        selected_data["publish_author"] = selected_data["publish_author"].map(id_map)

        return selected_data["publish_author"], selected_data["publish_time"].astype(np.int64) // 10 ** 11, len(
            all_writer)


class NewsBatch(Dataset):
    def __init__(self, original_dataset: NewsDataset, event_length=100):
        self.event_length = event_length
        self.dataset = original_dataset
        self.author_list, self.time_list, self.num_of_author = self.dataset.get_data()

    def __len__(self):
        return len(self.author_list) - self.event_length

    def __getitem__(self, index):
        return self.author_list[index:index + self.event_length], self.time_list[index:index + self.event_length]




if __name__ == '__main__':
    a = Dataset().get_data()
    print(1)
