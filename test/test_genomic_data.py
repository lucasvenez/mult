import unittest
import preprocessing


class TestGenomicData(unittest.TestCase):

    def test_read_genomic_data(self):

        df = preprocessing.read_genomic_data('../input/iss_status.tsv')

        self.assertEqual(df.head().shape[0], 5)

        self.assertTrue('MMRF_1014' in df.index.values)


    def test_extract_gene_ids(self):

        gene_ids = preprocessing.extract_all_gene_ids()

        self.assertEqual(len(gene_ids), 27147)
