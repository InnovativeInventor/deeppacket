#!/usr/bin/env python3
from scapy.all import *
import tensorflow as tf
import numpy as np
import sys
import glob
from tqdm import tqdm
import secrets
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, ConvLSTM2D
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM
from keras import backend as K
import matplotlib.pyplot as plt
import functools


## Debugging
# import tracemalloc
# tracemalloc.start()



PACKET_BYTES = 1600 # truncate/pad
PROCESSED_BYTES = 2000

def grab_data_keras(file):
    # TODO: Check if https and vpn pcap files exist
    raw = {
        "file": rdpcap(file),
    }
    file_buffer  = np.zeros((len(raw['file']), PACKET_BYTES), dtype="uint8")

    # TODO: figure out precise definition of "Raw" in the context of scapy
    raw_file_packets = 0
    for idx, packet in enumerate(raw['file']):
        if Raw in packet:
            # case 1: pad
            if len(packet[Raw].load) < PACKET_BYTES:
                print(packet[Raw].show) # Debugging
                buffer = np.frombuffer(packet[Raw].load, dtype="uint8")
                file_buffer[idx][:buffer.size] = buffer
            # case 2: truncate
            else:
                file_buffer[idx] = np.frombuffer(packet[Raw].load[:PACKET_BYTES], dtype="uint8")
            raw_file_packets += 1
    data = file_buffer[:raw_file_packets]
    
    # print(len(file_buffer))
    # print(file_buffer.shape)
    # print(data.shape)
    return data

@functools.lru_cache()
def grab_data_modified_keras(file):
    raw = rdpcap(file)
    data_array = []
    for idx, packet in enumerate(raw):
        if len(data_array) is not 200:
            if len(data_array) is 0:
                data_array.append(0)
            else:
                data_array.append(packet.time-prev_packet_time)
            
            data_array.append(len(packet))
            prev_packet_time = packet.time
    
    while len(data_array) < 200:
        data_array.append(0)
    
    numpy_array = np.asarray(data_array)
    return numpy_array
    

def data_yield(batch_size, https,vpn):
    """
    Should spit out a dictionary for the labels and a list of the training/eval data
    This code is messy, but will do.
    """
    folder_list = [https, vpn]
    while True:
        data = []
        label = []
        data_size = 0
        for i in range(batch_size):

            coin_flip = secrets.choice(folder_list)
            folder_files = glob.glob(coin_flip+"/*.pcap")
            data_unorganized = grab_data_modified_keras(secrets.choice(folder_files))
            data_swapped = np.swapaxes(np.expand_dims(data_unorganized, axis=0),0,1)
            data_flipped = np.swapaxes(data_swapped,1,0)
            data.append(data_flipped)
            x,y = data_flipped.shape
            data_size += data_flipped.size
            # for i in range(y):
            if coin_flip == https:
                # print("https")
                label.append(np.array([0]))
            else:
                # print("vpn")
                label.append(np.array([1]))

                ## Debugging
                # gc.collect

        label = np.concatenate(label, axis=0)
        label_expanded = np.expand_dims(label, axis=0)
        label_swapped = np.swapaxes(label_expanded, 0, 1)

        # print(data)
        stacked_data = np.asarray(data)
        final_data = np.swapaxes(stacked_data,1,2)
        # expanded_data = np.expand_dims(final_data, axis=0)
        # print(final_data.shape)
        yield final_data, label_swapped


def gather_data_keras():
    labels_train = {}
    data_train_https, labels_train_https = data_dictionary(sys.argv[1], "training", 0)
    data_train_vpn, labels_train_vpn = data_dictionary(sys.argv[2], "training", 1)
    labels_train.update(labels_train_https)
    labels_train.update(labels_train_vpn)
    data_train = data_train_https + data_train_vpn

    labels_eval = {}
    data_eval_https, labels_eval_https = data_dictionary(sys.argv[3], "eval", 0)
    data_eval_vpn, labels_eval_vpn = data_dictionary(sys.argv[4], "eval", 1)
    labels_eval.update(labels_eval_https)
    labels_eval.update(labels_eval_vpn)
    data_eval = data_eval_https + data_eval_vpn

    return labels_train, data_train, labels_eval, data_eval


def data_generator_train():
    labels_train, data_train, labels_eval, data_eval = gather_data_keras()
    random.shuffle(data_train)
    for _, data in enumerate(data_train):
        yield data, labels_train[data]
 

def data_generator_eval():
    labels_train, data_train, labels_eval, data_eval = gather_data_keras()
    random.shuffle(data_eval)
    for _, data in enumerate(data_eval):
        yield data, labels_eval[data]
 

def discriminate_keras():
    # for data, labels in data_yield(sys.argv[1],sys.argv[2]):
    #     pass
    #     # print(data.shape)
    model_keras()


def model_keras():
    K.set_image_dim_ordering('tf') 
    print(K.tensorflow_backend._get_available_gpus())
    batch_size = 15
    # From https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    model = Sequential()
    # model.add(LSTM(200, return_sequences=True, input_shape=(1600, 1)))
    # model.add(Flatten())
    # model.add(LSTM(32, return_sequences=True))
    
    model.add(Conv1D(32, (3), input_shape=(200, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    # model.add(Conv1D(32, (3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=(2)))

    # model.add(Conv1D(32, (3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=(2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.summary() # For debugging
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()
    history = model.fit_generator(
        data_yield(batch_size, sys.argv[1], sys.argv[2]),
        steps_per_epoch=4000 // batch_size,
        epochs=25,
        validation_data=data_yield(batch_size, sys.argv[3], sys.argv[4]),
        validation_steps=800 // batch_size,
        max_queue_size=2
    )
    model.save('first_try.h5')
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.gcf().clear()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")
    

if __name__ == "__main__":
    discriminate_keras()