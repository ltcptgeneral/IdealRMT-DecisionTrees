# Setup

Run `pip install -r requirements.txt`

Run `setup.sh`

# Tree Generation

## Download Dataset

Download the *September 22 2016* dataset (or others) from: https://iotanalytics.unsw.edu.au/iottraces.html#bib18tmc

Place these into the `data/tar` folder.

Run `extract_tars.sh` which will extract and place the `.pcap` files at the corresponding location inside `data/pcap`.

## Preprocessing Dataset

Run `extract_all_datasets.py` which will extract the data from each file in `data/pcap` and turn it into the corresponding `.csv` file inside `data/processed`. This will take a few minutes per file. Combine the data under `data/csv` using `combine_csv.py`. This will overwrite `data/combined/data.csv` which you can use for the decision tree.

## Training

Run `DecisionTree.ipynb`, the tree should be output in `tree`