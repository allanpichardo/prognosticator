#!/usr/bin/env bash

python yahooconverter.py ./csv/qcom_price.csv ./csv/qcom_quarterly.csv ./csv/test/test1.csv && 
python yahooconverter.py ./csv/xom_price.csv ./csv/xom_quarterly.csv ./csv/test/test2.csv

rm ./csv/test/combined.csv
python yahooconverter.py combine ./csv/test
