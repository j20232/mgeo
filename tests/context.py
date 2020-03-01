import unittest

from tests.test_linalg import LinalgTest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(LinalgTest))
    return suite
