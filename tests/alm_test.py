import unittest
from abc import ABC, abstractmethod

class ALMTest(unittest.TestCase, ABC):
    @classmethod
    def setUpClass(cls):
        super(ALMTest, cls).setUpClass()
        cls.model = LanguageModel.load()

    def setUp(self):
        self.model.reset()

    @abstractmethod
    def test_something(self):
        pass

class OpenAITest(AbstractTest):
    def test_something(self):
        pass
