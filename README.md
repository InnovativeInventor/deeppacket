## Building and Running
Build:
```
docker build -t deeppacket .
```

Train:
```
python3 discriminate_keras.py data/https data/openvpn data/https_eval data/openvpn_eval
```

Train in background:
```
nohup python3 discriminate_keras.py data/https data/openvpn data/https_eval data/openvpn_eval & disown
```

## Dependecies 
- Docker
- [tqdm](https://github.com/tqdm/tqdm)

Tested with Python 3.6.6

## Gathering data
Make sure you have the aws cli set up and access to the s3 bucket named `deeppacket.homelabs.space`. Run `scripts/grab_data.sh`

```
Note: This may be a requester-pays bucket in the future.
```

Split the http, https, and openvpn data into the training data and evaulation data (this can be randomly done).

