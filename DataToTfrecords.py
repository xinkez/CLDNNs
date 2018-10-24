import tensorflow as tf
import librosa
import numpy as np

def ReadDataList(data_path):
    file = []
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f]
        for line in lines:
            file.append(line)
    return file


def save_tfrecords(data, label, desfile, writer):

    features = tf.train.Features(
        feature={
            "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.astype(np.float32).tostring()])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
    )
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)

def ReadData(file):
    file = ReadDataList(file)
    with tf.python_io.TFRecordWriter("./test.tfrecords") as writer:
        t = 0
        for i in range(len(file)):
            file_name, lab = file[i].strip().split('\t')
            if file_name != '.DS_Store':
                try:
                    frame, sr = librosa.load(file_name)
                    frame = librosa.util.frame(frame, frame_length=560, hop_length=400)
                except:
                    continue
                frame = frame.T
                print(frame.shape)
                label = int(lab)
                #print(label)
                #print(frame)
                save_tfrecords(frame, label, desfile="./data.tfrecords", writer=writer)
                print(file_name + " SAVED")
                t = t + 1
                print(t)

if __name__ == '__main__':
    ReadData('/Users/Roy/PycharmProjects/CLDNN/audio/test.list')