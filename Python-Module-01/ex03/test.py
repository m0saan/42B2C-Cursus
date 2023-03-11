import unittest
from generator import generator
class TestGenerator(unittest.TestCase):

    def test_valid_text(self):
        text = "Le Lorem Ipsum est simplement du faux texte."
        expected_output = ["Le", "Lorem", "Ipsum", "est", "simplement", "du", "faux", "texte."]
        self.assertListEqual(list(generator(text, sep=" ")), expected_output)

    def test_shuffle_option(self):
        text = "Le Lorem Ipsum est simplement du faux texte."
        output = list(generator(text, sep=" ", option="shuffle"))
        self.assertEqual(len(output), len(text.split(" ")))
        self.assertNotEqual(output, text.split(" "))

    def test_ordered_option(self):
        text = "Le Lorem Ipsum est simplement du faux texte."
        expected_output = ["Ipsum", "Le", "Lorem", "du", "est", "faux", "simplement", "texte."]
        self.assertListEqual(list(generator(text, sep=" ", option="ordered")), expected_output)

    def test_unique_option(self):
        text = "Lorem Ipsum Lorem Ipsum"
        expected_output = ["Lorem", "Ipsum"]
        self.assertListEqual(list(generator(text, sep=" ", option="unique")), expected_output)

    def test_invalid_text(self):
        text = 1.0
        self.assertEqual(list(generator(text, sep=".")), ["ERROR"])

    def test_invalid_option(self):
        text = "Le Lorem Ipsum est simplement du faux texte."
        self.assertEqual(['ERROR'], list(generator(text, sep=" ", option="invalid_option")))

if __name__ == '__main__':
    unittest.main()
