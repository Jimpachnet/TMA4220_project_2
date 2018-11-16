import unittest
import numpy as np

from project_2.infrastructure.affine_transformation_3d import AffineTransformation3D

class TestCode(unittest.TestCase):
    """
    Useful test cases
    """

    def test_affine_transformation(self):
        """
        Tests the affine transformation
        :return:
        """

        affine_trafo = AffineTransformation3D()

        # Reference
        x0 = (0, 0, 0)
        x1 = (1, 0, 0)
        x2 = (0, 1, 0)
        x3 = (0, 0, 1)
        self.assertEqual(affine_trafo.get_determinant(x0, x1, x2, x3), 1)

        # Invalid simplex
        x1 = x0
        self.assertEqual(affine_trafo.get_determinant(x0, x1, x2, x3), 0)

if __name__ == '__main__':
    print("Starting unittest...")
    unittest.main()