import pandas as pd
import numpy as np
import os
import progressbar
from random import shuffle


class Config:
    # note: the Excel dataset follows this order
    nodes = ["Pelvis", "Neck", "Head", "Right Shoulder", "Right Upper Arm",
             "Right Forearm", "Right Hand", "Left Shoulder", "Left Upper Arm",
             "Left Forearm", "Left Hand", "Right Upper Leg", "Right Lower Leg",
             "Right Foot", "Right Toe", "Left Upper Leg", "Left Lower Leg",
             "Left Foot", "Left Toe"]

    label_fp = "dataset/label/Wisconsin Gait Scale.xlsx"
    labels = pd.read_excel(label_fp)

    spine_segment = ["L5", "L3", "T12", "T8"]


# Version 1: only extract position as features

class Person:
    def __init__(self, filepath, sheets=None, need="some"):
        """
        :param filepath:
            fp , e.g., "dataset/data/xxxxx.xlsx"
        :param sheets:
            IMU generate multiple features (sheets)
            Currently, only use position information
        """
        self.max_frames = 0
        self.min_frames = 10000
        self.labels = list()
        self.pos_features = list()
        if sheets is None:
            sheets = []
        self.name = None
        self.fp = filepath
        self.sheets = sheets
        if need == "some":
            self.extract()
        else:
            self.extract(time_split=1)

    def extract(self, ignore_spine=True, time_split=4):
        filename = self.fp.split('/')[-1].split('.')[0]
        self.name = filename.split(' ')[0]  # in this way, we can directly match the label
        pos_sheet = pd.read_excel(self.fp, sheet_name="Segment Position")
        cyc_sheet = pd.read_excel(self.fp, sheet_name="Markers")
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
        if ignore_spine:
            for seg in Config.spine_segment:
                for coord in ["x", "y", "z"]:
                    drop_labels.append(seg + " " + coord)
        pos_sheet = pos_sheet.drop(drop_labels, axis=1)

        rowId = int(self.name[1:])
        labels = Config.labels.loc[rowId + 1][2:]  # filter timestamp and name
        universe_labels = [int(labels[t][-1]) for t in range(len(labels))]

        for start in range(0, len(cyc_sheet), time_split):
            if start + 4 < len(cyc_sheet):
                end = start + time_split
                pos_features = pos_sheet.to_numpy()  # in shape [frames, joints * 3(3D coords)]
                interval_start = cyc_sheet['Frame'][start]
                interval_end = cyc_sheet['Frame'][end]
                pos_features = pos_features[interval_start:interval_end]
                self.max_frames = max(pos_features.shape[0], self.max_frames)
                self.min_frames = min(self.min_frames, pos_features.shape[0])
                # print(pos_featuers.shape)
                pos_features = np.reshape(pos_features, (pos_features.shape[0], -1, 3))
                pos_features = np.transpose(pos_features, (2, 0, 1))
                self.pos_features.append(pos_features)
                self.labels.append(universe_labels)


class Dataset:
    def __init__(self, dspath="./dataset/data"):
        """
        :param dspath:
            dataset path
        what we are going to do here is to find the maximum time frames of the dataset,
        then do interpolation to achieve batch training.  (firstly simply use linear xx)

        another thing is to split the dataset into train & test set.
        """
        self.train_ptr = 0
        self.test_ptr = 0
        self.maxFrame = 0
        self.minFrame = 10000
        self.train_data = list()
        self.train_label = list()
        self.test_data = list()
        self.test_label = list()
        self.dspath = dspath
        np.random.seed(1)
        total_files = os.listdir(dspath)
        total_len = len(total_files)
        train_num = int(total_len * 0.7)
        test_num = total_len - train_num
        train_set = np.random.choice(total_len, train_num, replace=False)
        print("Start loading files:")
        p = progressbar.ProgressBar()
        test_set = list()
        for i in p(range(total_len)):
            if i not in train_set:
                test_set.append(i)
                self.load_person(total_files[i], train=False)
            else:
                self.load_person(total_files[i])
        test_set = np.array(test_set)
        actual_train_num = len(self.train_data)
        actual_test_num = len(self.test_data)
        print("Finish loading.")
        print("Minimum Frame of one gait cycle is {}".format(self.minFrame))  # 59
        print("Maximum Frame of one gait cycle is {}".format(self.maxFrame))  # 859
        self.padding()

    def load_person(self, fp, train=True):
        p = Person(os.path.join(self.dspath, fp), need="some")
        self.maxFrame = max(self.maxFrame, p.max_frames)  # for interpolation
        self.minFrame = min(self.minFrame, p.min_frames)
        if train:
            for features, labels in zip(p.pos_features, p.labels):
                self.train_data.append(features)
                self.train_label.append(labels)
        else:
            for features, labels in zip(p.pos_features, p.labels):
                self.test_data.append(features)
                self.test_label.append(labels)

    def padding(self):
        pass

    def load_batch_data(self, train=True):
        pass

    def load_data(self, train=True):
        """
        for item 4 Wisconsin Gait Scale Item
        """
        data = None
        label = None
        if train:
            data = self.train_data[self.train_ptr]
            label = self.train_label[self.train_ptr][4] - 1
            self.train_ptr += 1
            if self.train_ptr == len(self.train_data):
                self.train_ptr = 0
        else:
            data = self.test_data[self.test_ptr]
            label = self.test_label[self.test_ptr][4] - 1
            self.test_ptr += 1
            if self.test_ptr == len(self.test_data):
                self.test_ptr = 0
        return data, label

    def shuffle(self):
        """
        :return:
                don't try to directly shuffle on "self.train_data" since we may not get the correct label.
                Thus, shuffle the index and use a new container to replace the original one

                don't try to directly operate on original container to avoid data overwriting or data missing
        """
        indexs = list([i for i in range(len(self.train_data))])
        shuffle(indexs)
        new_train_data = list()
        new_label_data = list()
        for j in indexs:
            new_train_data.append(self.train_data[j])
            new_label_data.append(self.train_label[j])
        self.train_label = new_label_data
        self.train_data = new_train_data
        self.train_ptr = 0

# code for debugging
# a = Person("dataset/data/S6 C-002.xlsx")
# D = Dataset()
