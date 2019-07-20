import os
import csv
import shutil
import pandas as pd


def cls_train_data(csv_path, img_dir, save_root):
    df = pd.read_csv(csv_path, header=0)
    names = df[["name"]].values
    labels = df[[" label"]].values
    names = names.reshape((1, -1))
    names = names.tolist()[0]
    labels = labels.reshape((1, -1))
    labels = labels.tolist()[0]

    for i, item in enumerate(names):
        img_path = os.path.join(img_dir, item)
        if not os.path.exists(img_path):
            print(img_path)
            continue
        label = str(labels[i])

        save_dir = os.path.join(save_root, label)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, item)

        shutil.move(img_path, save_path)

if __name__ == '__main__':
    csv_path = '/work/competitions/TinyMind/train_face_value_label.csv'
    img_dir = '/work/competitions/TinyMind/train_data'
    save_root = '/work/competitions/TinyMind/PreTrainData'
    cls_train_data(csv_path, img_dir, save_root)
