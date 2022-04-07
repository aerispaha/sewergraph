import unittest

from sewergraph import hhcalculations


class TestHHCalculations(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_mannings_capacity(self):

        q = hhcalculations.mannings_capacity(diameter=12, slope=0.02)
        self.assertAlmostEqual(q, 4.3764, places=4)
