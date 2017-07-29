from asl_test_model_selectors import TestSelectors
import unittest
suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
unittest.TextTestRunner().run(suite)