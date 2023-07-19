import pandas as pd


class Config:
    train_rate = 0.7

    # take "Center of Mass" as additional joint
    # so when this's True, please append this joint to the nodes
    use_CoM = True

    # note: the Excel dataset follows this order
    nodes = ["Pelvis", "Neck", "Head", "Right Shoulder", "Right Upper Arm",
             "Right Forearm", "Right Hand", "Left Shoulder", "Left Upper Arm",
             "Left Forearm", "Left Hand", "Right Upper Leg", "Right Lower Leg",
             "Right Foot", "Right Toe", "Left Upper Leg", "Left Lower Leg",
             "Left Foot", "Left Toe"]

    label_fp = "dataset/label/Wisconsin Gait Scale.xlsx"
    labels = pd.read_excel(label_fp)

    label_dict = {"item1": "Use of hand-held gait aid",
                  "item2": "Stance time on impaired side",
                  "item3": "Step length of unaffected side",
                  "item4": "Weight shift to the affected side with or without gait aid",
                  "item5": "Stance width",
                  "item6": "Guardedness",
                  "item7": "Hip extension of the affected leg",
                  "item8": "External rotation during initial swing",
                  "item9": "Circumduction at mid-swing",
                  "item10": "Hip hiking at mid-swing",
                  "item11": "Knee flexion from toe off to mid-swing",
                  "item12": "Toe clearance",
                  "item13": "Pelvic rotation at terminal swing",
                  "item14": "Initial foot contact"
                  }

    classes = [5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3]

    segment_sheets = ["Segment Orientation - Euler", "Segment Position", "Segment Velocity",
                      "Segment Acceleration", "Segment Angular Velocity", "Segment Angular Acceleration", ]

    # corresponding to the segment sheets
    # by default, we only use position information as features,
    # so the input features should be in shape [3, frames, joints = (nodes num)]
    # with one additional sheet used, there'll be 3 more channels.
    segment_sheet_idx = [1]

    spine_segment = ["L5", "L3", "T12", "T8"]

    # since the spine segments are estimated by IMU, I firstly decide not to use it
    ignore_spine = True

    # extract gait cycles method
    # option:   4   or   1
    # use 1 to generate more gait cycles
    time_split = 4

    # batch training
    batch_size = 4

    # test which item
    item = 3
