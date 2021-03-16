from unittest import TestCase
import torch

class TestTorchCuda(TestCase):
    def test_torch_gpu(self):
        self.assertTrue(torch.cuda.is_available())
