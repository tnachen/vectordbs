from vectordbs import factory
import unittest

class TestPineconeIntegration(unittest.TestCase):
    def test_pinecone_simple():
        pinecone_db = factory.get_datastore("pinecone")

