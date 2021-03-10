from .predict import predict
import argparse
import sys
import pkg_resources
from .transformer import make_model
import torch

# TODO -- load weights & model properly here for testing
weights=pkg_resources.resource_filename(__name__,"soltrannet_aqsol_trained.weights")

'''
tmp=make_model()
use_cuda = torch.cuda.is_available()
if use_cuda:
    device=torch.device("cuda")
    tmp.load_state_dict(torch.load(weights))
    tmp.to(device)
else:
    device=torch.device('cpu')
    tmp.load_state_dict(torch.load(weights,map_location=device))
self.model=tmp
'''

def run(self):
    parser=argparse.ArgumentParser(description="Run SolTranNet aqueous solubility predictor")
    parser.add_argument('-i','--input',required=True,nargs='?',type=argparse.FileType('r'),default=sys.stdin,help='PATH to the file containing the SMILES you wish to use. Assumes the content is 1 SMILE per line.')
    parser.add_argument('-o','--output',nargs='?',type=argparse.FileType('w'),default=sys.stdout,help='Name of the output file.')
    parser.add_argument('--batchsize',default=32,help='Batch size for the data loader. Defaults to 32.')
    args=parser.parse_args()

    smiles=[x.rstrip() for x in args.input.readlines()]
    args.input.close()
    predictions=predict(smiles,batch_size=args.batchsize)
    for pred, smi, warn in predictions:
        args.output.write(f'{smi} {pred} {warn}\n')
    '''
    if args.output:
        with open(args.output,'w') as outfile:
            for pred, smi, warn in predictions:
                outfile.write(f'{smi} {pred} {warn}\n')
    else:
        for pred, smi, warn in predictions:
            print(pred,smi,warn)
    '''