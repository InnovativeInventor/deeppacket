"""
Neural network that discriminates between VPN packets and HTTPS packets.
The paper "Deep Packet" (Lotfollahi 2017, https://arxiv.org/abs/1709.02656) is a useful introduction to feeding packets into NNs.
It suggests using Conv1D layers
We're not currently using an autoencoder, but we are padding and truncating packets as suggested in the paper.
"""
from scapy.all import *
import tensorflow as tf
import numpy as np
import sys
import glob
from tqdm import tqdm
import secrets

PACKET_BYTES = 1600 # truncate/pad
PROCESSED_BYTES = 2000
# based on https://www.tensorflow.org/tutorials/estimators/cnn
def cnn_model(features, labels, mode): # Needs to be named properly
    packets = features
    input_layer = tf.reshape(packets, [-1, PROCESSED_BYTES])
    input_layer.shape()
    # pool1 = tf.layers.max_pooling1d(inputs=input_layer, pool_size=2, strides=2) # Added this to go from rank 2 > rank 3 (may not be necessary)

    third_rank_input_layer = tf.expand_dims(input_layer, axis=0)

    conv1 = tf.layers.conv1d(
        third_rank_input_layer,
        32,
        5,
        padding='same',
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)
    conv2 = tf.layers.conv1d(
        pool1,
        64,
        5,
        padding='same',
        activation=tf.nn.relu
    )

    # conv1 = tf.nn.convolution(
    #     input_layer,
    #     32,
    #     'same',
    #     activation=tf.nn.relu
    # )
    # pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)
    # conv2 = tf.nn.convolution(
    #     pool1,
    #     64,
    #     'same',
    #     activation=tf.nn.relu
    # )

    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)

    dimension_size = int(128) # Debug
    print("Dimension size: " + str(dimension_size)) # Debug
    pool2_flat = tf.reshape(pool2, [-1, dimension_size]) # May need to be PACKET_BYTES
    # pool2_flat = tf.contrib.layers.flatten(
    #     pool2,
    # )

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=dense, units=1)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode = mode, loss=loss, train_op=train_op)


def grab_data(https, vpn):
    # TODO: Check if https and vpn pcap files exist
    raw = {
        "https": rdpcap(https),
        "vpn": rdpcap(vpn)
    }
    https_buffer = np.zeros((len(raw['https']), PACKET_BYTES), dtype="uint8")
    vpn_buffer   = np.zeros((len(raw['vpn']), PACKET_BYTES), dtype="uint8")

    # TODO: figure out precise definition of "Raw" in the context of scapy
    raw_https_packets = 0
    for idx, packet in enumerate(raw['https']):
        if Raw in packet:
            # case 1: pad
            if len(packet[Raw].load) < PACKET_BYTES:
                # print(packet[Raw].load) # Debugging
                buffer = np.frombuffer(packet[Raw].load, dtype="uint8")
                https_buffer[idx][:buffer.size] = buffer
            # case 2: truncate
            else:
                https_buffer[idx] = np.frombuffer(packet[Raw].load[:PACKET_BYTES], dtype="uint8")
            raw_https_packets += 1

    raw_vpn_packets = 0
    for idx, packet in enumerate(raw['vpn']):
        if Raw in packet:
            # case 1: pad
            if len(packet[Raw].load) < PACKET_BYTES:
                buffer = np.frombuffer(packet[Raw].load, dtype="uint8")
                vpn_buffer[idx][:buffer.size] = buffer
            # case 2: truncate
            else:
                vpn_buffer[idx] = np.frombuffer(packet[Raw].load[:PACKET_BYTES], dtype="uint8")
            raw_vpn_packets += 1

    train  = np.concatenate((https_buffer[:raw_https_packets], vpn_buffer[:raw_vpn_packets]), axis=0) # Training data
    labels = np.concatenate((np.zeros([raw_https_packets]), np.ones([raw_vpn_packets])), axis=0) # Labels
    return train, labels


def combine_data(https, vpn, purpose):
    https_files = glob.glob(https+"/*.pcap")
    vpn_files = glob.glob(vpn+"/*.pcap")
    length = min(len(https_files), len(vpn_files))

    print("Preprocessing " + purpose + " data")
    for i in tqdm(range(length)):
        train_data, labels_data = grab_data(https_files[i],vpn_files[i])
        if i == 0:
            train = train_data
            labels = labels_data
        else:
            train = np.concatenate((train, train_data), axis=0)
            labels = np.concatenate((labels, labels_data), axis=0)
    
    return train.view('float64'), labels.view('float64')


# def data_generator(https, vpn, purpose):
#     https_files = glob.glob(https+"/*.pcap")
#     vpn_files = glob.glob(vpn+"/*.pcap")

#     train_data, labels_data = grab_data(secrets.choice(https_files), secrets.choice(vpn_files))
    
#     return train_data.view('float64'), labels_data.view('float64')


def discriminate_tf():
    # Logging
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, 
        every_n_iter=50
    )

    data_train, labels_train = combine_data(sys.argv[1], sys.argv[2], "training")
    data_eval, labels_eval = combine_data(sys.argv[3], sys.argv[4], "evaluation")
    classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir="model_tmp")
    print(type(data_train))

    train_in = tf.estimator.inputs.numpy_input_fn(
        data_train,
        y=labels_train,
        batch_size=100, # Might not be right size
        num_epochs=None,
        shuffle=True
    )

    classifier.train(
        input_fn=train_in,
        steps=10000,
        hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data_eval},
        y=labels_eval,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    # PICK UP HERE


if __name__ == "__main__":
    if len(sys.argv) is not 6 or not 5:
        """
        Usage:
            python3 discriminate.py [https pcap folder] [vpn pcap folder] [https_eval pcap folder] [vpn_eval pcap folder] [backend]

            Backends:
                - keras
                - tensorflow
        """
        raise ValueError("Not enough arguments provided!")

    if sys.argv[5] == "keras":
        print("Using Keras")
        # discriminate_keras()
    else:
        print("Using TensorFlow")
        discriminate_tf()