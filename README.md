## Building and Running
Build:
```
docker build -t deeppacket .
```

Run:
```
python3 discriminate.py regular openvpn regular_eval openvpn_eval
```

## Dependecies 
- Docker
- [tqdm](https://github.com/tqdm/tqdm)

Tested with Python 3.6.6

## Training in background with nohup
```
nohup python3 discriminate_keras.py https openvpn https_eval openvpn_eval & disown
```