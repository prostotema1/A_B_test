import unittest

from bootstrap import bootstrap


class TestBootstrap(unittest.TestCase):
    values_a1 = [4, 4, 4, 4, 4]
    values_b1 = [3, 3, 3, 3, 3]

    values_a2 = [5, 5, 5, 5, 5]
    values_b2 = [5, 5, 5, 4.9, 4.9]

    def setUp(self):
        self.bs = bootstrap

    def test_bs_true(self):
        self.assertEqual(self.bs(self.values_a1, self.values_b1)[0], True)

    def test_bs_false(self):
        self.assertEqual(self.bs(self.values_a2, self.values_b2)[0], False)