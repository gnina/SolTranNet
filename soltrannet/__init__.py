from .predict import predict
import argparse

def run():
    parser=argparse.ArgumentParser(description="Run SolTranNet aqueous solubility predictor")
    parser.add_argument('-i','--input',required=True,help='PATH to the file containing the SMILES you wish to use. Assumes the content is 1 SMILE per line.')
    parser.add_argument('-o','--output',required=True,help='Name of the output file.')
    parser.add_argument('--batchsize',default=32,help='Batch size for the data loader. Defaults to 32.')
    args=parser.parse_args()

    smiles=[x.rstrip() for x in open(args.input).readlines()]
    predictions=predict(smiles,batch_size=batchsize)
    with open(args.output,'w') as outfile:
        for pred, smi, warn in predictions:
            outfile.write(f'{smi} {pred} {warn}\n')