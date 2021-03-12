import argparse
from unittest import TestCase
from soltrannet import _run
import io

class TestCommandLine(TestCase):
    def test_command_line(self):
        with io.StringIO() as buf:
            correct='c1ccccc1,-1.053,\nc1ccccc1 .ignore,-1.053,\nCn1cnc2n(C)c(=O)n(C)c(=O)c12,-1.132,\n[Zn+2],-6.882,Other-typed Atom(s) Detected Prediction less reliable\n[Na+].[Cl-],-0.169,Salt Other-typed Atom(s) Detected Prediction less reliable\n'

            args=argparse.Namespace(input=['c1ccccc1','c1ccccc1 .ignore','Cn1cnc2n(C)c(=O)n(C)c(=O)c12','[Zn+2]','[Na+].[Cl-]'],
                                                                                             output=buf,batchsize=5, cpus=0,cpu_predict=None)
            _run(args)
            self.assertEqual(buf.getvalue(),correct)
