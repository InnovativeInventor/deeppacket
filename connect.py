import requests
from bs4 import BeautifulSoup
import validators
import subprocess
import secrets
import sys
import os
import time

"""
Usage: python3 connect.py [protocol]
Supported protocols:
    - OpenVPN
    - https
    - http
"""


def main():
    if len(sys.argv) < 2:
        print("Error invalid blank argument")
        raise ValueError("Arguments are 'openvpn' or a folder name.")

    print("Option selected: " + sys.argv[1])
    
    if "http" in sys.argv[1]:
        protocol = "http://"
    else:
        protocol = "https://"

    links = random_links(protocol)
    time.sleep(2)


    if "openvpn" in sys.argv[1]:
        process = tcpdump()
        time.sleep(5)
        connect_openvpn()
        time.sleep(25)
        

    if not os.path.isdir(sys.argv[1]):
        os.makedirs(sys.argv[1])

    if len(links) >= 1:
        process = tcpdump()
        visit_links(links,protocol)
        
    else:
        main()
    
    process.terminate()
    openvpn_terminate()
    os._exit(0)

def openvpn_terminate():
    subprocess.run(["tc", "qdisc", "del", "dev", "eth0", "root"])
    # subprocess.run(["ping", "-c", "10", "google.com"]) # Debug

def connect_openvpn():
    """
    Connects to a remote server using OpenVPN in a subprocess. This needs to have a ovpn file named client.ovpn in the same directory.
    """
    delay = str(secrets.randbelow(80)) + "ms"
    # subprocess.run(["ping", "-c", "10", "google.com"]) # Debug
    print("Delay: " + delay) # Debug
    subprocess.run(["tc", "qdisc", "add", "dev", "eth0", "root", "netem", "delay", delay, "1ms", "distribution", "normal"])
    time.sleep(2)
    # subprocess.run(["ping", "-c", "10", "google.com"]) # Debug
    subprocess.Popen(["openvpn", "--config", "client.ovpn"])


def tcpdump():
    process = subprocess.Popen(["tcpdump", "-n", "-U", "-w", sys.argv[1] + "/" + secrets.token_hex(8) + ".pcap"])
    return process


def random_links(protocol):
    response = requests.get("https://en.wikipedia.org/wiki/Special:Random")
    soup = BeautifulSoup(response.content, "html.parser")

    ext_links = []

    links = soup.find_all('a', href=True)
    for result in links:
        url = result['href']
        if validators.url(url) and protocol and "wiki" not in url:
            try:
                requests.get(url, timeout=3)
                ext_links.append(url)
            except:
                pass

    return ext_links
    
def visit_links(links, protocol):
    for each_link in links:
        if protocol in each_link:
            print(each_link)
            requests.get(each_link) # Only needs to be the length of a vpn handshake


if __name__ == "__main__":
    main()
