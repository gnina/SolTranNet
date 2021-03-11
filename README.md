# SolTranNet
The official implementation of SolTranNet.

[comment]: # (TODO: Add an html reference to above when the paper is published.)

SolTranNet is an optimized fork of the [Molecule Attention Transformer](https://github.com/ardigen/MAT), whose original paper can be found [here](https://arxiv.org/abs/2002.08264).

## TODO
~~1) Get package version running~~

~~2) Write predict function~~

~~3) remove pandas dependency~~

~~4) remove training script~~

~~5) organize repo and source code~~

~~6) Add liscence?~~

~~7) Test CPU version of in python works correctly~~

~~8) Create/test command line version~~

9) make pip installable
10) Finish README

## Requirements
 - PyTorch 1.7+
 - RDKit 2020.03.1dev1

### Soft Requirements
 - CUDA 10.1 or CUDA 10.2

We heavily suggest installing CUDA and compiling PyTorch with it enabled to have faster models.

## Installation
First, install [RDKit](https://github.com/rdkit/rdkit). Installation instructions are available [here](https://github.com/rdkit/rdkit/blob/master/Docs/Book/Install.md)

After RDKit has finished installing, you can install SolTranNet via pip:
```
pip install soltrannet
```

## Usage

### Command line tool
Upon successful pip installation, a command line tool will be installed.

To generate the predictions for SMILES provided in `my_smiles.txt` and store them into `my_output.txt`:
```
soltrannet my_smiles.txt my_output.txt
```

### In a Python environment
Soltrannet also supports integration in a python3 environment

```python
import soltrannet as stn
my_smiles=["c1ccccc1","Cn1cnc2n(C)c(=O)n(C)c(=O)c12","[Zn+2]","[Na+].[Cl-]"]
predictions=list(stn.predict(my_smiles))
```

## Help
Please [subscribe to our slack team](https://join.slack.com/t/gninacnn/shared_invite/enQtNTY3ODk2ODk5OTU5LTkzMjY1ZTE3YjJlZmIxOWI2OTU3Y2RlMTIyYmM2YmFmYTU1NTk5ZTBmMjUwMGRhYzk1ZjY5N2E4Y2I5YWU5YWI).

[comment]: # (TODO: Add a BibTex reference to the paper when published.)