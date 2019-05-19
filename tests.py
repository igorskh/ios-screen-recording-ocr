import unittest

import text_helpers

class TestTextHelpersMethods(unittest.TestCase):
    def test_distance(self):
        test_cases = [
            ("test_string", "test string", 1),
            ("test_string", "teststring", 1),
            ("test_strings", "teststring", 2),
            ("test_string", "test_string", 0)
        ]
        for t in test_cases:
            res = text_helpers.minimum_edit_distance(t[0], t[1])
            self.assertEqual(res, t[-1])

    def test_in_range(self):
        test_cases = [
            (15, (0, 10), False),
            (15, (0, 15), False),
            (15, (0, 16), True),
            (-15, (-20, 10), True),
            (-22, (-20, 10), False)
        ]
        for t in test_cases:
            res = text_helpers.is_val_in_range(t[0], t[1])
            self.assertEqual(res, t[-1])

    def test_distance_replace(self):
        test_cases = [
            ("test_string", "test string", 1, "test_string"),
            ("test_string", "teststring", 1, "test_string"),
            ("test_strings", "teststring", 1, "teststring"),
            ("test_string", "test_string", 1, "test_string"),
            ("test_string", "node_string", 1, "node_string"),
            ("test_string", "node_string", 4, "test_string")
        ]
        for t in test_cases:
            res = text_helpers.check_distance(t[1], t[0], t[2])
            self.assertEqual(res, t[-1])


    def test_simple_replace(self):

        REPLACE_RULES = {
            "1,-1": ["i", "[", "]", "l", "7", "?", "t"],
            "1,0": ["i", "[", "]", "l", "7", "?", "t"],
            "q,": ["qg","qq","gg","gq"]
        }
        test_cases = [
            ("testi", "1est1"),
            ("test?", "1est1"),
            ("?test", "1tes1"),
            ("]tesn", "1tesn"),
            ("qg", "q"),
            ("qgqg", "q"),
            ("qgqgqg", "qq")
        ]
        for t in test_cases:
            res = text_helpers.check_replace(t[0], REPLACE_RULES)
            self.assertEqual(res, t[-1])

if __name__ == '__main__':
    unittest.main()