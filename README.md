# Setup

Run `pip install -r requirements.txt`

# Tree Generation

## Download Dataset

Download the *September 22 2016* dataset from: https://iotanalytics.unsw.edu.au/iottraces.html#bib18tmc

Rename the file as data.pcap

## Preprocessing Dataset

Run `ExtractDataset.ipynb`, this will take a few minutes

## Training

Run `DecisionTree.ipynb`, the tree should be output in `tree`