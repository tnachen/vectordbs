import unittest
from vectordbs import factory

class TestFactory(unittest.TestCase):
    def test_factory():
        pinecone = factory.get_datastore("pinecone")
        assert pinecone