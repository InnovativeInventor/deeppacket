FROM python:latest

LABEL maintainer="InnovativeInventor"

RUN apt-get update && apt-get install -y \
    tcpdump \
    nano \
    openssl \
    curl \
    python3-pip \
    python3 \
    openvpn \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install beautifulsoup4 requests validators
