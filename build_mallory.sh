# !/bin/bash
cd /host/mediator
apt-get update
apt-get install -y sudo
sudo apt-get install -y musl-tools
sudo apt-get install -y musl-dev
apt-get install -y openssh-client
cd /host/mediator && rustup target add x86_64-unknown-linux-musl
RUSTFLAGS="-C target-cpu=native" cargo build --release --target=x86_64-unknown-linux-musl --target-dir /mediator-target
mkdir -p /root/.ssh
ssh-keyscan n1 n2 n3 n4 n5 >> /root/.ssh/known_hosts