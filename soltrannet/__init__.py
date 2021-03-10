from .predict import predict

def _run():
    import argparse
    parser=argparse.ArgumentParser(description="Run SolTranNet aqueous solubility predictor")
    parser.add_argument('-i','--input',nargs='?',type=argparse.FileType('r'),default=sys.stdin,help='PATH to the file containing the SMILES you wish to use. Assumes the content is 1 SMILE per line.')
    parser.add_argument('-o','--output',nargs='?',type=argparse.FileType('w'),default=sys.stdout,help='Name of the output file.')
    parser.add_argument('--batchsize',default=32,help='Batch size for the data loader. Defaults to 32.')
    args=parser.parse_args()

    smiles=[x.rstrip() for x in args.input.readlines()]
    args.input.close()
    predictions=predict(smiles,batch_size=args.batchsize)
    for pred, smi, warn in predictions:
        args.output.write(f'{smi} {pred} {warn}\n')
