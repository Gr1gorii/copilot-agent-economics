#!/bin/bash
# about 3.2 GB
mkdir -p data/raw
cd data/raw
for d in 01 02 03 04 05 06 07; do
  curl -L -O https://github.com/Azure/AzurePublicDataset/releases/download/ghcp-coding-agent-2026/date.2026-06-$d.tar.gz
done
