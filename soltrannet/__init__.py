from .predict import predict 
import argparse
import sys, multiprocessing
import torch

def _parse_args():
    parser=argparse.ArgumentParser(description="Run SolTranNet aqueous solubility predictor")
    parser.add_argument('input',nargs='?',type=argparse.FileType('r'),default=sys.stdin,help='PATH to the file containing the SMILES you wish to use. Assumes the content is 1 SMILE per line.')
    parser.add_argument('output',nargs='?',type=argparse.FileType('w'),default=sys.stdout,help='Name of the output file.')
    parser.add_argument('--batchsize',default=32,type=int,help='Batch size for the data loader. Defaults to 32.')
    parser.add_argument('--cpus',default=multiprocessing.cpu_count(),type=int,help='Number of CPU cores to use. Defaults to use all available cores. Pass 0 to only run on 1 CPU.')

    args=parser.parse_args()

    return args

def _run(args):

    smiles=[x.rstrip() for x in args.input]
    args.input.close()
    predictions=predict(smiles,batch_size=args.batchsize,num_workers=args.cpus)
    for pred, smi, warn in predictions:
        args.output.write(f'{smi},{pred:.3f},{warn}\n')

