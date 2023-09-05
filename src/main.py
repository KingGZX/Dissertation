import torch

from loadata import Dataset
from train import *
from baselines.ST_GCN_baseline import ST_GCN
from baselines.Uniformer_baseline import Uniformer
from config import Config

if __name__ == "__main__":
    cfg = Config()
    # person = Person(filepath="dataset/data/patient/S37 陳永棠 (L)-002.xlsx", cfg=cfg)

    dataset = Dataset(cfg, dataset_path="./test_dst/data")

    model = Uniformer(in_channels=cfg.in_channels)

    batch_train(model, dataset, epochs=10, model_name="Uniformer", cfg=cfg)
