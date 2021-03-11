from unittest import TestCase
from soltrannet.data_utils import load_data_from_smiles
from soltrannet.predict import check_smile

class TestWarning(TestCase):
    def test_working_warnings(self):
        smiles=["c1ccccc1","Cn1cnc2n(C)c(=O)n(C)c(=O)c12","c1ccccc1 .ignore","[Zn+2]","[Na+].[Cl-]"]
        features=load_data_from_smiles(smiles)
        node_features=[x[0] for x in features]
        warnings=[]
        for smi, f in zip(smiles,node_features):
            warnings.append(check_smile(smi,f))

        self.assertTrue(warnings==['','','','Other-typed Atom(s) Detected Prediction less reliable','Salt Other-typed Atom(s) Detected Prediction less reliable'])
