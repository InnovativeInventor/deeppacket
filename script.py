import subprocess
import os
import sys
from tqdm import tqdm
import time
# docker build . -t deeppacket

"""
Usage: python3 script.py [number of pcap files] [folder to save in]
"""
reboot_frequency = 50
server_ip = "35.174.172.204"
if len(sys.argv) < 3:
    print("Not enough arguments")
    raise ValueError("Usage: python3 script.py [number of pcap files] [folder to save in]")

for i in tqdm(range(int(sys.argv[1]))):
    # if "openvpn" in sys.argv[2]:
    subprocess.call(["docker", "run", "--rm", "-it", "-v", os.getcwd() + ":/deeppacket", "-w", "/deeppacket", "--cap-add", "NET_ADMIN", "deeppacket", "python3", "connect.py", sys.argv[2]])
    if i % reboot_frequency == 0:
        "Rebooting server!"
        subprocess.run(["ssh", "-i", "server.pem", "ubuntu@35.174.172.204", "sudo", "reboot"])
        time.sleep(120)
    # else:
        # subprocess.call(["docker", "run", "--rm", "-it", "-v", os.getcwd() + ":/deeppacket", "-w", "/deeppacket", "deeppacket", "python3", "connect.py", sys.argv[2]])
    print()
