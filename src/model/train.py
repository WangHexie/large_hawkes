from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from src.data.dataset import NewsDataset, train_test_split, NewsBatch
from src.model.hawkes import MFHawkes


class HawkesTrainer(pl.LightningModule):
    def __init__(self, hawkes_model, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = hawkes_model

    def forward(self, identity, times):
        # use forward for inference/predictions
        embedding = self.backbone(identity, times)
        return embedding

    def training_step(self, batch, batch_idx):
        y_hat = self(*batch)
        loss = y_hat.mean()
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(*batch)
        loss = y_hat.mean()
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        y_hat = self(*batch)
        loss = y_hat.mean()
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = HawkesTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    original_dataset = NewsDataset()

    data_and_info = original_dataset.get_data()
    train_authors, train_times, test_authors, test_times = train_test_split(*data_and_info[:2])

    train_data = NewsBatch(train_authors, train_times)
    test_data = NewsBatch(test_authors, test_times)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    # val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    # test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = HawkesTrainer(MFHawkes(data_and_info[-1], args.hidden_dim), args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # # ------------
    # # testing
    # # ------------
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == '__main__':
    cli_main()
