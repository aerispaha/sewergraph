import unittest

from sewergraph import hhcalculations


class TestHHCalculations(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_mannings_capacity_circle(self):

        q = hhcalculations.mannings_capacity(diameter=12, slope=0.02)
        self.assertAlmostEqual(q, 4.3764, places=4)

    def test_mannings_capacity_box(self):
        q = hhcalculations.mannings_capacity(diameter=None, slope=0.02, height=12, width=12, shape='BOX')
        self.assertAlmostEqual(q, 5.5723, places=4)
