#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from labels import mac_to_label
from tqdm import tqdm
import os

ROOT       = Path(__file__).resolve().parent
PCAP_DIR   = ROOT / "data" / "pcap"
CSV_DIR    = ROOT / "data" / "processed"
CSV_DIR.mkdir(parents=True, exist_ok=True)

BATCH = 100_000   # packets per chunk

from scapy.all import rdpcap


def process_pcap(pcap_path: str, csv_path: str) -> None:
    all_packets = rdpcap(pcap_path)

    print("rdpcap done", flush=True)
    results = []
    for packet in tqdm(all_packets):
        size = len(packet)
        try:
            proto = packet.proto
        except AttributeError:
            proto = 0
        try:
            sport = packet.sport
            dport = packet.dport
        except AttributeError:
            sport = 0
            dport = 0

        proto = int(proto)
        sport = int(sport)
        dport = int(dport)

        if "Ether" in packet:
            eth_dst = packet["Ether"].dst
            if eth_dst in mac_to_label:
                classification = mac_to_label[eth_dst]
            else:
                classification = "other"
        else:
            classification = "other"

        metric = [proto,sport,dport,classification]
        results.append(metric)
    results = (np.array(results)).T

    # store the features in the dataframe
    dataframe = pd.DataFrame({'protocl':results[0],'src':results[1],'dst':results[2],'classfication':results[3]})
    columns = ['protocl','src','dst','classfication']

    # save the dataframe to the csv file, if not exsit, create one.
    if os.path.exists(csv_path):
        dataframe.to_csv(csv_path,index=False,sep=',',mode='a',columns = columns, header=False)
    else:
        dataframe.to_csv(csv_path,index=False,sep=',',columns = columns)
        
    print("Done")



def main() -> None:
    for pcap in sorted(PCAP_DIR.rglob("*.pcap")):
        rel_csv = pcap.relative_to(PCAP_DIR).with_suffix(".csv")
        csv_path = CSV_DIR / rel_csv
        if csv_path.exists():
            print(f"Skip {rel_csv} (CSV exists)")
            continue
        print(f"Processing {rel_csv}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        process_pcap(str(pcap), str(csv_path))

if __name__ == "__main__":
    main()
