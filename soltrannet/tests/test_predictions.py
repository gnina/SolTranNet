from unittest import TestCase
import soltrannet as stn
import torch
import numpy as np

class TestPredictions(TestCase):
    def test_preds(self):

        def helper(a,b):
            a_array=np.array([x[0] for x in a])
            b_array=np.array([x[0] for x in b])

            first = np.any(np.isclose(a_array,b_array))
            second = [x[1] for x in a] == [x[1] for x in b]
            third = [x[2] for x in a] == [x[2] for x in b]

            return first and second and third

        smiles=["c1ccccc1","[Zn+2]","[Na+].[Cl-]"]
        correct=[(-1.052748441696167, 'c1ccccc1', ''),(-6.881845474243164,'[Zn+2]','Other-typed Atom(s) Detected Prediction less reliable'),(-0.16869020462036133,'[Na+].[Cl-]','Salt Other-typed Atom(s) Detected Prediction less reliable')]
        
        test_cpu=list(stn.predict(smiles,device='cpu'))
        self.assertTrue(helper(correct, test_cpu))

        if torch.cuda.is_available():
            test_gpu=list(stn.predict(smiles,device='cuda'))
            self.assertTrue(helper(correct, test_gpu))
