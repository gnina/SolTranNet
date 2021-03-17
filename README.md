# SolTranNet
The official implementation of SolTranNet.

[comment]: # (TODO: Add an html reference to above when the paper is published.)

SolTranNet is an optimized fork of the [Molecule Attention Transformer](https://github.com/ardigen/MAT), whose original paper can be found [here](https://arxiv.org/abs/2002.08264).

## Requirements
 - Python 3.6+
 - PyTorch 1.7+
 - RDKit 2017.09.1+
 - pathlib 1.0+

### Soft Requirements
 - CUDA 10.1, 10.2, or 11.1

We heavily suggest installing CUDA and compiling PyTorch with it enabled to have faster models.

You can see installation instructions for CUDA [here](https://developer.nvidia.com/cuda-toolkit-archive).

You can see installation instructions for PyTorch [here](https://pytorch.org/).

## Installation
Tested on Ubuntu 18.04.5, Ubuntu 20.04.2, Debian 10, Fedora 33, CentOS 8.3.2011, Windows 10, and Ubuntu 20.04.2 subsystem for Windows 10

First, install [RDKit](https://github.com/rdkit/rdkit). Installation instructions are available [here](https://github.com/rdkit/rdkit/blob/master/Docs/Book/Install.md)

After RDKit has finished installing, you can install SolTranNet via pip:
```
python3 -m pip install soltrannet
```
NOTE: This installation method often mismatches installation of PyTorch for enabling CUDA if it needs to install PyTorch as a dependency.

If you wish to do a more careful installation:
```
python3 -m pip install --install-option test soltrannet
```
This will run our unit tests to ensure that GPU-enabled torch was setup correctly, and the proper functioning of SolTranNet as a command line tool and within a python environment.

## Usage

### Command line tool
Upon successful pip installation, a command line tool will be installed.

To generate the predictions for SMILES provided in `my_smiles.txt` and store them into `my_output.txt`:
```
soltrannet my_smiles.txt my_output.txt
```

You can see all of the options available for the command line tool:
```
usage: soltrannet [-h] [--batchsize BATCHSIZE] [--cpus CPUS] [--cpu_predict] [input] [output]

Run SolTranNet aqueous solubility predictor

positional arguments:
  input                 PATH to the file containing the SMILES you wish to
                        use. Assumes the content is 1 SMILE per line.

  output                Name of the output file. Defaults to stdout.

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
                        Batch size for the data loader. Defaults to 32.

  --cpus CPUS           Number of CPU cores to use for the data loader.
                        Defaults to use all available cores. Pass 0 to only
                        run on 1 CPU.

  --cpu_predict         Flag to force the predictions to be made on only the
                        CPU. Default behavior is to use GPU if available.

```

### In a Python environment
Soltrannet also supports integration in a python3 environment

```python
import soltrannet as stn
my_smiles=["c1ccccc1","c1ccccc1 .ignore","Cn1cnc2n(C)c(=O)n(C)c(=O)c12","[Zn+2]","[Na+].[Cl-]"]
predictions=list(stn.predict(my_smiles))
```

## Help
Please [subscribe to our slack team](https://join.slack.com/t/gninacnn/shared_invite/enQtNTY3ODk2ODk5OTU5LTkzMjY1ZTE3YjJlZmIxOWI2OTU3Y2RlMTIyYmM2YmFmYTU1NTk5ZTBmMjUwMGRhYzk1ZjY5N2E4Y2I5YWU5YWI).

[comment]: # (TODO: Add a BibTex reference to the paper when published.)
