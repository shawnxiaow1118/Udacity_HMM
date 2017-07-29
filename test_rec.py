from asl_test_recognizer import TestRecognize
import unittest

suite = unittest.TestLoader().loadTestsFromModule(TestRecognize())
unittest.TextTestRunner().run(suite)