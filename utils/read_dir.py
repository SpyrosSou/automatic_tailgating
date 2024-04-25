import os
import sys


class ReadDir():
    def __init__(self,
                 base_dir,
                 subset='testing',
                 labels=True
                 ):
        # Todo: set local base dir
        # self.base_dir = '/home/user/PycharmProjects/dataset/kitti/data_object_image_2/'
        self.base_dir = base_dir

        # if use kitti training data for train/val evaluation
        if subset == 'training':
            self.label_dir = os.path.join(self.base_dir, subset, 'label_2/')
            self.image_dir = os.path.join(self.base_dir, subset, 'image_2/')
            self.calib_dir = os.path.join(self.base_dir, subset, 'calib/')

        # if use raw data
        if subset == 'testing':
            self.tracklet_drive = os.path.join(self.base_dir, subset)
            self.image_dir = os.path.join(self.tracklet_drive, 'image_2/')
            self.calib_dir = os.path.join(self.tracklet_drive, 'calib/')
            self.prediction_dir = os.path.join(self.base_dir, "smoke_predictions/predictions")
            # Check if labels are available based on flag passed as argument
            if labels:
                self.label_dir = os.path.join(self.tracklet_drive, 'label_2/')
            # If labels are not available, just import the training ones, they won't be used anyway
            else:
                self.label_dir = os.path.join(self.base_dir, "training", 'label_2/')



if __name__ == '__main__':
    dir = ReadDir(subset='training')
    dir_ = ReadDir(subset='tracklet',
                    tracklet_date='2011_09_06',
                    tracklet_file='2011_09_26_drive_0084_sync')
    print(dir.image_dir)
    print(dir_.image_dir)