from loadata import Dataset
from train import train
from baselines.st_gcn_baseline import ST_GCN
import argparse

if __name__ == "__main__":
    dataset = Dataset()
    # model = ST_GCN(3, 3)
    # train(model, dataset)