import unittest
import scipy.stats as sps

from Main import *


class Tests(unittest.TestCase):
    def testMann_Withney_true(self):
        sample1 = getSampleWithUniformDistribuition(1, 5, 200)
        sample2 = getSampleWithUniformDistribuition(1, 5, 100)
        self.assertEquals(Mann_whitneyu(sample1, sample2), sps.mannwhitneyu(sample1, sample2))

    def test_mann_withney_statictik(self):
        sample1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sample2 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.assertEqual(Mann_whitneyu(sample1, sample2)[0], 12.5)

    def test_z_Proportion(self):
        rez = z_test_proportions(165, 300, 196, 400)
        self.assertTrue(rez[0] - 1.57195 < 0)
