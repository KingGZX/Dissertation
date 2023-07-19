from loadata import Dataset
from train import train
from baselines.st_gcn_baseline import ST_GCN
from baselines.TCN_baseline import TCN
from config import Config

if __name__ == "__main__":
    cfg = Config()

    dataset = Dataset(cfg)

    # model = TCN(t_kernel=9, time_steps=dataset.maxFrame, num_classes=3)

    # model = ST_GCN(3, 3)

    # train(model, dataset, item=cfg.item, epochs=40, baseline="TCN")