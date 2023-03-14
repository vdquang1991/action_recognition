import os
import csv

def get_classes_name(root_path):
    classes = sorted(os.listdir(root_path))
    return classes

def get_data(root_path, classes):
    data = []
    for c in classes:
        folders = os.listdir(os.path.join(root_path, c))
        for f in folders:
            length_file = len(os.listdir(os.path.join(root_path,c,f)))
            data.append([root_path, c, f, length_file])
    return data

root_path = 'train'
classes = get_classes_name(root_path)
data = get_data(root_path, classes)

with open("train.csv", "w") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(data)
