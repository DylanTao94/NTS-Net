import torch.utils.data as data
import os
import sys
import random
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE

def make_dataset(root, source, frame_rate):
    if not os.path.exists(source):
        print("Setting file %s for haa500_basketball dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                video_path = os.path.join(root, line_info[0])
                duration = int(line_info[1])
                # TODO: add sample rate here
                label = int(line_info[2])
                for i in range(duration):
                    item = (video_path, i+1, label)
                    clips.append(item)
    return clips

def ReadImg(path, frame_index, new_height, new_width, is_color, name_pattern):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR
    frame_name = name_pattern % (frame_index)
    frame_path = path + "/" + frame_name
    cv_img_origin = cv2.imread(frame_path, cv_read_flag)
    if cv_img_origin is None:
       print("Could not load file %s" % (frame_path))
       sys.exit()
       # TODO: error handling here
    if new_width > 0 and new_height > 0:
        # use OpenCV3, use OpenCV2.4.13 may have error
        cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
    else:
        cv_img = cv_img_origin
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return np.concatenate([cv_img], axis=2)


def data_transform(items, phase):
    items_processed = []
    if phase == "train":
        for item in items:
            if len(item.shape) == 2:  # if only 1 img in this batch
                item = np.stack([item] * 3, 2)
            item = Image.fromarray(item, mode='RGB')
            item = transforms.Resize((600, 600), Image.BILINEAR)(item)
            item = transforms.RandomCrop(INPUT_SIZE)(item)
            item = transforms.RandomHorizontalFlip()(item)
            item = transforms.ToTensor()(item)
            item = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(item)
            items_processed.append(item)

    elif phase == "val":
        for item in items:
            if len(item.shape) == 2:
                item = np.stack([item] * 3, 2)
            item = Image.fromarray(item, mode='RGB')
            item = transforms.Resize((600, 600), Image.BILINEAR)(item)
            item = transforms.CenterCrop(INPUT_SIZE)(item)
            item = transforms.ToTensor()(item)
            item = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(item)
            items_processed.append(item)

    return items_processed


class haa500_basketball(data.Dataset):

    def __init__(self,
                 root, # dataset root dir
                 source, # train_split_file
                 phase,
                 modality,
                 is_color=True,
                 frame_rate=1,
                 new_width=0,
                 new_height=0,):

        data_items = make_dataset(root, source, frame_rate)

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality
        self.frame_rate = frame_rate
        self.data_items = data_items
        self.is_color = is_color
        self.new_width = new_width
        self.new_height = new_height
        if self.modality == "rgb":
            self.name_pattern = "img_%05d.jpg"
        elif self.modality == "flow":
            self.name_pattern = ["flow_x_%05d.jpg", "flow_y_%05d.jpg"]

    def __getitem__(self, index):
        path, frame_index, target = self.data_items[index]
        items = []
        if self.modality == "rgb":
            frame = ReadImg(path, frame_index, self.new_width, self.new_width, self.is_color, self.name_pattern)
            items.append(frame)
        elif self.modality == "flow":
            flow_x = ReadImg(path, frame_index, self.new_width, self.new_width, self.is_color, self.name_pattern[0])
            flow_y = ReadImg(path, frame_index, self.new_width, self.new_width, self.is_color, self.name_pattern[1])
            items.append(flow_x)
            items.append(flow_y)
        else:
            print("No such modality %s" % (self.modality))

        items_processed = data_transform(items, self.phase)

        return items_processed, target

    def __len__(self):
        return len(self.data_items)
