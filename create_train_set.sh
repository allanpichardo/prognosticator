#!/usr/bin/env bash

python yahooconverter.py ./csv/amd_price.csv ./csv/amd_quarterly.csv ./csv/train/train1.csv && 
python yahooconverter.py ./csv/goog_price.csv ./csv/goog_quarterly.csv ./csv/train/train2.csv && 
python yahooconverter.py ./csv/msft_price.csv ./csv/msft_quarterly.csv ./csv/train/train3.csv && 
python yahooconverter.py ./csv/mu_price.csv ./csv/mu_quarterly.csv ./csv/train/train4.csv && 
python yahooconverter.py ./csv/nke_price.csv ./csv/nke_quarterly.csv ./csv/train/train5.csv && 
python yahooconverter.py ./csv/noc_price.csv ./csv/noc_quarterly.csv ./csv/train/train6.csv && 
python yahooconverter.py ./csv/nvda_price.csv ./csv/nvda_quarterly.csv ./csv/train/train7.csv && 
python yahooconverter.py ./csv/pfe_price.csv ./csv/pfe_quarterly.csv ./csv/train/train8.csv

rm ./csv/train/combined.csv
python yahooconverter.py combine ./csv/train
