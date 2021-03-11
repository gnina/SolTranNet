from unittest import TestCase
from soltrannet.data_utils import construct_loader_from_smiles
import torch

class TestLoading(TestCase):
    def test_loader(self):
        smiles=["c1ccccc1","[Zn+2]","[Na+].[Cl-]"]
        data_loader=construct_loader_from_smiles(smiles,batch_size=len(smiles),num_workers=0)
        check=list(data_loader)

        correct=[torch.tensor([[[0., 0., 0., 0., 0., 0., 0.],[0., 1., 1., 0., 0., 0., 1.],[0., 1., 1., 1., 0., 0., 0.],[0., 0., 1., 1., 1., 0., 0.],[0., 0., 0., 1., 1., 1., 0.],[0., 0., 0., 0., 1., 1., 1.],[0., 1., 0., 0., 0., 1., 1.]],
                               [[0., 0., 0., 0., 0., 0., 0.],[0., 1., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0.]],
                               [[0., 0., 0., 0., 0., 0., 0.],[0., 1., 0., 0., 0., 0., 0.],[0., 0., 1., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0.]]]),
                torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1.],
                               [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1.],
                               [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1.],
                               [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1.],
                               [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1.],
                               [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1.]],
                              [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.,0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
                              [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.,0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]),
                ["c1ccccc1","[Zn+2]","[Na+].[Cl-]"],
                [0,1,2]
        ]

        self.assertTrue(torch.all(torch.eq(check[0][0],correct[0])) and torch.all(torch.eq(check[0][1],correct[1])) and check[0][2]==correct[2] and check[0][3]==correct[3])