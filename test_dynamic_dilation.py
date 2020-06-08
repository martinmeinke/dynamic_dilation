import torch
import torch.nn as nn
import torch.nn.functional as F

import unittest

from dynamic_dilation import DynamicDilation

class DynamicDilationTest(unittest.TestCase):
    def test_run(self):
        """
        simply test if fwd pass works...
        """
        in_range_image = torch.ones(1, 1, 10, 10)
        in_range_image[:, :, 2:5, 3:8] = 20.1
        in_size = (5, 5)
        in_features = torch.ones(1, 1, 5, 5)

        print(in_range_image.shape)

        dd = DynamicDilation(1, 1, min_dil_range=50,
                             max_dil_range=0, smallest_dil=1, largest_dil=5)
        out = dd(in_features, in_range_image)


if __name__ == '__main__':
    unittest.main(verbosity=2)
