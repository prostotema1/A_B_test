import unittest

from bootstrap import bootstrap


class TestBootstrap(unittest.TestCase):
    values_a = [4, 4, 4, 4, 4]
    values_b = [3, 3, 3, 3, 3]

    def setUp(self):
        self.bs = bootstrap
    def test_bs(self):
        self.assertEqual(self.bs(self.values_a, self.values_b), True)