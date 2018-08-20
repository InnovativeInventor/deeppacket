source activate tensorflow_p36
pip install tqdm scapy
nohup python3 discriminate_keras.py data/https data/openvpn data/https_eval data/openvpn_eval & disown