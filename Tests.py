import unittest
import scipy.stats as sps

from Main import *


class Tests(unittest.TestCase):

    def test_t_Test(self):
        sample1 = get_sample_with_uniform_distribuition(-1, 2, 1000)
        sample2 = get_sample_with_uniform_distribuition(-100, 200, 1000)
        self.assertFalse(t_test(sample1, sample2) < 0.05)
    def testMann_Withney_true(self):
        sample1 = get_sample_with_uniform_distribuition(1, 5, 200)
        sample2 = get_sample_with_uniform_distribuition(1, 5, 100)
        self.assertFalse(mann_whitneyu(sample1, sample2) < 0.05)

    def test_mann_withney_statictik(self):
        sample1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sample2 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.assertTrue(mann_whitneyu(sample1, sample2) < 0.05)

    def test_z_Proportion(self):
        rez = z_test_proportions(165, 300, 196, 400)
        self.assertFalse(rez)

    def test_z_Proportion2(self):
        rez = z_test_proportions(100, 200, 10, 500)
        self.assertTrue(rez)

    def test_t_Test_vs_manna_whitney(self):
        rez = compare_t_test_and_mann_whitney(10000)
        self.assertTrue(rez)