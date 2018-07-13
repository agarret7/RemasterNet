# Standard Modules
import unittest

# External Modules
import numpy as np

# Internal Modules
from util import *

np.random.seed(0)

class BasicTests(unittest.TestCase):
    """ Basic test cases """

    def test_foo(self):
        self.assertTrue(True)
        self.assertEqual(1, 1)

    def test_bar(self):
        self.assertTrue(np.isclose(1.0, 1.00001))
        self.assertFalse(np.isclose(1.0, 1.0001))

if __name__ == "__main__":
    unittest.main()
