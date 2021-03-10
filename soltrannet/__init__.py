from .predict import predict 
import argparse
import sys, multiprocessing
import torch

def run():
    parser=argparse.ArgumentParser(description="Run SolTranNet aqueous solubility predictor")
    parser.add_argument('input',nargs='?',type=argparse.FileType('r'),default=sys.stdin,help='PATH to the file containing the SMILES you wish to use. Assumes the content is 1 SMILE per line.')
    parser.add_argument('output',nargs='?',type=argparse.FileType('w'),default=sys.stdout,help='Name of the output file.')
    parser.add_argument('--batchsize',default=64,type=int,help='Batch size for the data loader. Defaults to 64.')
    parser.add_argument('--cpus',default=multiprocessing.cpu_count(),type=int,help='Number of CPU cores to use.')

    args=parser.parse_args()

    smiles=[x.rstrip() for x in args.input]
    args.input.close()
    predictions=predict(smiles,batch_size=args.batchsize,num_workers=args.cpus)
    for pred, smi, warn in predictions:
        args.output.write(f'{smi} {pred:.3f} {warn}\n')

