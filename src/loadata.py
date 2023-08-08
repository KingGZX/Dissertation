import pandas as pd
import numpy as np
import os
import progressbar
from config import Config
from random import shuffle


# Version 1: only extract position as features
class Person:
    def __init__(self, filepath, cfg: Config):
        """
        :param filepath:
                 fp , e.g., "dataset/data/xxxxx.xlsx"
        """
        self.use_CoM = False
        self.mass_center_features = None
        self.max_frames = 0
        self.min_frames = 10000
        self.frames = 0
        self.labels = list()
        self.features = list()
        self.sheet = list()
        self.name = None
        self.fp = filepath
        self.cfg = cfg
        self.extract(cfg.time_split)

    def extract(self, time_split):
        filename = self.fp.split('/')[-1].split('.')[0]
        self.name = filename.split(' ')[0]  # in this way, we can directly match the label
        cyc_sheet = pd.read_excel(self.fp, sheet_name="Markers")  # recording gait cycles

        """
        now I want to take all the valid gait cycles to enrich my dataset

        # extract the first cycle end
        interval_start = cyc_sheet['Frame'][0 + 4]
        # L/R Toe Off -> L/R Heel Strike -> R/L Toe Off -> R/L Heel Strike -> L/R Toe Off
        interval_end = cyc_sheet['Frame'][0 + 8]
        # above code helps extract the second cycle
        """
        # print(pos_sheet.columns)
        drop_labels = ["Frame"]
        if self.cfg.ignore_spine:
            for seg in self.cfg.spine_segment:
                for coord in ["x", "y", "z"]:
                    drop_labels.append(seg + " " + coord)

        for sheet_id in self.cfg.segment_sheet_idx:
            sheet_name = self.cfg.segment_sheets[sheet_id]
            sheet = pd.read_excel(self.fp, sheet_name=sheet_name)
            sheet = sheet.drop(drop_labels, axis=1)
            self.sheet.append(sheet)

        rowId = int(self.name[1:])  # filter 'S'
        labels = self.cfg.labels.loc[rowId - 1][2:]  # filter timestamp and name
        universe_labels = [int(labels[t][-1]) for t in range(len(labels))]

        if self.cfg.use_CoM:
            try:
                # some of the exported files don't have this working sheet at first
                mass_center_sheet = pd.read_excel(self.fp, sheet_name="Center of Mass")
                mass_center_sheet = mass_center_sheet.drop(["Frame"], axis=1)
                self.mass_center_features = mass_center_sheet.to_numpy()  # [frames, joints * 3]
                self.use_CoM = True
            except:
                # accidentally find that there are some files don't contain this worksheet
                self.use_CoM = False
                print("{} doesn't have sheet Center of Mass\n".format(self.fp))

        for start in range(0, len(cyc_sheet), time_split):
            if start + time_split < len(cyc_sheet):
                end = start + time_split
                interval_start = cyc_sheet['Frame'][start]
                interval_end = cyc_sheet['Frame'][end]
                self.max_frames = max(interval_end - interval_start, self.max_frames)
                self.min_frames = min(self.min_frames, interval_end - interval_start)
                self.frames += interval_end - interval_start

                features = None

                for sheet in self.sheet:
                    # these are all segment relevant features, positions、velocity、acceleration and so on
                    # the joints are the same, so we're just enlarging the channels
                    x_sheet_features = sheet.to_numpy()  # in shape [frames, joints * 3(3D coords)]
                    x_sheet_features = x_sheet_features[interval_start:interval_end]
                    # print(pos_featuers.shape)
                    x_sheet_features = np.reshape(x_sheet_features, (x_sheet_features.shape[0], -1, 3))
                    if features is None:
                        features = np.transpose(x_sheet_features, (2, 0, 1))  # [channel, frames, joints]
                    else:
                        x_sheet_features = np.transpose(x_sheet_features, (2, 0, 1))
                        features = np.concatenate([features, x_sheet_features], axis=0)

                if self.use_CoM:
                    com_features = self.mass_center_features[interval_start:interval_end]
                    com_features = np.reshape(com_features, (com_features.shape[0], 1, -1))
                    com_features = np.transpose(com_features, (2, 0, 1))
                    channel, frame, joint = com_features.shape
                    channel_1 = features.shape[0]
                    if channel != channel_1:
                        if channel >= channel_1:
                            com_features = com_features[:channel_1]
                        else:
                            # use zero-padding to enlarge the channels
                            zero_blocks = np.zeros((channel_1 - channel, frame, joint))
                            com_features = np.concatenate([com_features, zero_blocks], axis=0)

                    # take "Center of Mass" as a new joint
                    features = np.concatenate([features, com_features], axis=-1)

                self.features.append(features)
                self.labels.append(universe_labels)


class Dataset:
    def __init__(self, cfg: Config, dataset_path="./dataset/data"):
        """
        :param cfg:
            a Config object
        :param dataset_path:
            dataset path

        what we are going to do here is to find the maximum time frames of the dataset,
        then do interpolation to achieve batch training.  (firstly simply use linear xx)

        another thing is to split the dataset into train & test set.
        """
        self.train_ptr = 0
        self.test_ptr = 0
        # since the frames spent on each gait cycle is different between different people
        # the following frames variable is recorded for statistics and padding
        self.maxFrame = 0
        self.minFrame = 10000
        self.total_frames = 0
        self.train_data = list()
        self.train_label = list()
        self.test_data = list()
        self.test_label = list()

        # for debugging overfitting problem
        self.train_name = list()
        self.test_name = list()
        self.train_cycle_index = list()
        self.test_cycle_index = list()

        self.dspath = dataset_path
        self.cfg = cfg

        if self.cfg.use_CoM:
            self.cfg.nodes.append("Center of Mass")

        np.random.seed(1)

        # since the dataset is imbalanced, if we don't load it in this way.
        # the model may only be trained on the patient data
        categories = ["healthy", "patient"]
        # categories = ["patient"]

        for category in categories:
            path = os.path.join(self.dspath, category)
            total_files = os.listdir(path)
            total_len = len(total_files)
            train_num = int(total_len * self.cfg.train_rate)
            train_set = np.random.choice(total_len, train_num, replace=False)
            print("Start loading {} files:".format(category))
            p = progressbar.ProgressBar()
            for i in p(range(total_len)):
                fp = os.path.join(path, total_files[i])
                if i not in train_set:
                    self.load_person(fp, train=False)
                else:
                    self.load_person(fp)

        actual_train_num = len(self.train_data)
        actual_test_num = len(self.test_data)
        actual_num = actual_test_num + actual_train_num

        self.train_batches = actual_train_num / self.cfg.batch_size
        self.test_batches = actual_test_num / self.cfg.batch_size

        print("Finish loading.")
        print("Train Set: {}, Test Set: {}".format(len(self.train_data), len(self.test_data)))
        print("Minimum Frame of one gait cycle is {}".format(self.minFrame))  # 49
        print("Maximum Frame of one gait cycle is {}".format(self.maxFrame))  # 859
        print("Average Frame of one gait cycle is {}".format(self.total_frames / actual_num))  # 115

        # self.zero_padding()

    def load_person(self, fp, train=True):
        p = Person(fp, self.cfg)
        self.maxFrame = max(self.maxFrame, p.max_frames)  # for interpolation
        self.minFrame = min(self.minFrame, p.min_frames)
        self.total_frames += p.frames

        cycles = len(p.features)

        if train:
            for features, labels in zip(p.features, p.labels):
                self.train_data.append(features)
                self.train_label.append(labels)
            for i in range(cycles):
                self.train_name.append(p.name)
                self.train_cycle_index.append(i + 1)
        else:
            for features, labels in zip(p.features, p.labels):
                self.test_data.append(features)
                self.test_label.append(labels)
            for i in range(cycles):
                self.test_name.append(p.name)
                self.test_cycle_index.append(i + 1)

    def zero_padding(self):
        """
        use the most naive-zero padding first
        :return:
        """
        for i in range(len(self.train_data)):
            channel, frames, joints = self.train_data[i].shape
            if frames < self.maxFrame:
                zeros_pad = np.zeros((channel, self.maxFrame - frames, joints))
                self.train_data[i] = np.concatenate([self.train_data[i], zeros_pad], axis=1)

        for j in range(len(self.test_data)):
            channel, frames, joints = self.test_data[j].shape
            if frames < self.maxFrame:
                zeros_pad = np.zeros((channel, self.maxFrame - frames, joints))
                self.test_data[j] = np.concatenate([self.test_data[j], zeros_pad], axis=1)

    def load_batch_data(self, train=True):
        data = None
        label = None
        item = self.cfg.item
        if train:
            if (self.train_ptr + 1) < self.train_batches:
                pass
        pass

    def load_data(self, train=True):
        """
        one gait cycle once

        "item is in the config file , means which item in the Wisconsin Gait Scale we want to predict"

        since initially it's just a one-head model which can only output one label at once.
        """
        data = None
        label = list()
        items = self.cfg.item

        # for tackling overfitting problem
        patient_name = None
        cycle_index = None

        if train:
            patient_name = self.train_name[self.train_ptr]
            cycle_index = self.train_cycle_index[self.train_ptr]
            data = self.train_data[self.train_ptr]
            for item in items:
                label.append(self.train_label[self.train_ptr][item - 1] - 1)
            self.train_ptr += 1
            if self.train_ptr == len(self.train_data):
                self.train_ptr = 0
        else:
            patient_name = self.test_name[self.test_ptr]
            cycle_index = self.test_cycle_index[self.test_ptr]
            data = self.test_data[self.test_ptr]
            for item in items:
                label.append(self.test_label[self.test_ptr][item - 1] - 1)
            self.test_ptr += 1
            if self.test_ptr == len(self.test_data):
                self.test_ptr = 0

        if len(label) == 1:
            label = label[0]

        return data, label, patient_name, cycle_index

    def shuffle(self):
        """
        :return:
                don't try to directly shuffle on "self.train_data" since we may not get the correct label.
                Thus, shuffle the index and use a new container to replace the original one

                don't try to directly operate on source container to avoid data overwriting or data missing
        """
        indexs = list([i for i in range(len(self.train_data))])
        shuffle(indexs)
        new_train_data = list()
        new_label_data = list()

        new_train_name = list()
        new_train_cindex = list()

        for j in indexs:
            new_train_data.append(self.train_data[j])
            new_label_data.append(self.train_label[j])

            new_train_name.append(self.train_name[j])
            new_train_cindex.append(self.train_cycle_index[j])

        self.train_label = new_label_data
        self.train_data = new_train_data

        self.train_name = new_train_name
        self.train_cycle_index = new_train_cindex

        self.train_ptr = 0

# code for debugging
# a = Person("dataset/data/S6 C-002.xlsx")
# D = Dataset()
