# coding=utf-8
import numpy as np
import librosa
import random


def shuffle(data_path):

    file = []
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f]
        for line in lines:
            file.append(line)
    random.shuffle(file)

    return file


def next_batch(file, batch_size, batch_count):
    frames = []
    labels = []
    for i in range(batch_size):
        file_name, lab = file[i+batch_count].strip().split('\t')
        if (file_name!= ".DS_Store"):
            frame, sr = librosa.load(file_name)
            frame = librosa.util.frame(frame, frame_length=560, hop_length=400)
            frames.append(frame)
            label_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            label = label_init
            label[int(lab)] = 1
            labels.append(label)
    raw_frame = np.array(frames, float)
    raw_labels = np.array(labels, float)
    print(raw_labels)

    return raw_frame, raw_labels


if __name__ == '__main__':
    name = shuffle('/Users/Roy/PycharmProjects/CLDNN/audio/trainer.list')
    next_batch(name, 10, 1)
