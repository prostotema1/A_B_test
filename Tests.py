import unittest
import scipy.stats as sps

from Main import *


class Tests(unittest.TestCase):

    def test_t_Test(self):
        sample1 = getSampleWithUniformDistribuition(-1,2,1000)
        sample2 = getSampleWithUniformDistribuition(-100,200,1000)
        self.assertFalse(tTest(sample1,sample2) < 0.05)
    def testMann_Withney_true(self):
        sample1 = getSampleWithUniformDistribuition(1, 5, 200)
        sample2 = getSampleWithUniformDistribuition(1, 5, 100)
        self.assertFalse(Mann_whitneyu(sample1, sample2) < 0.05)

    def test_mann_withney_statictik(self):
        sample1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sample2 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.assertTrue(Mann_whitneyu(sample1, sample2) < 0.05)

    def test_z_Proportion(self):
        rez = z_test_proportions(165, 300, 196, 400)
        self.assertFalse(rez)

    def test_z_Proportion2(self):
        rez = z_test_proportions(100, 200, 10, 500)
        self.assertTrue(rez)

    def test_t_Test_vs_manna_whitney(self):
        rez = compare_tTest_and_Mann_Whitney(1000)
        self.assertTrue(rez)