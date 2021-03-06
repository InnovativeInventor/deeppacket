from keras.models import load_model
import sys
from scapy.all import *
import numpy as np

## Usage python3 deploy_discriminate.py [model location] [pcap file]
## Example python3 deploy_discriminate.py try_1.h5 ../../openvpn/9ceb86bcee0f50cd.pcap
model = load_model(sys.argv[1])


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

def run_prediction():
    formatted_data = grab_data_modified_keras(sys.argv[2])
    print(formatted_data.shape)
    sized_data = np.expand_dims(formatted_data, axis=1)
    final_data = np.expand_dims(sized_data, axis=0)
    print(sized_data.shape)
    print(final_data.shape)
    prediction = model.predict(final_data)
    print(prediction)

if __name__ == "__main__":
    run_prediction()