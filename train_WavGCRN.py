import os
import random
import numpy as np
import torch
# import setproctitle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='WavGCRN',help='model')
parser.add_argument('--data',type=str,default='METR-LA',help='dataset')
args = parser.parse_args()

model = args.model
data = args.data
# setproctitle.setproctitle(model + '_' + data + "@qqian")

random_seed = np.random.seed(42)

def main():
    if model == 'WavGCRN':
        if data == 'METR-LA':
            run = 'python ./methods/WavGCRN/train2.py  --adj_data ./data/sensor_graph/adj_mx.pkl --data ./data/METR-LA --num_nodes 207 --runs 1  --epochs 400 --print_every 1 --batch_size 64 --tolerance 700  --cl_decay_steps 4000 --expid DGCRN_metrla --device cuda:0'
            os.system(run)
        elif data == 'PEMS-BAY':
            run = 'python ./methods/WavGCRN/trainp.py --adj_data ./data/sensor_graph/adj_mx_bay.pkl --data ./data/PEMS-BAY --num_nodes 325 --runs 1 --epochs 250 --print_every 1 --batch_size 64 --tolerance 700 --expid DGCRN_pemsbay  --cl_decay_steps 5500 --rnn_size 96 --device cuda:0'
            os.system(run)

if __name__ == "__main__":
    main()
