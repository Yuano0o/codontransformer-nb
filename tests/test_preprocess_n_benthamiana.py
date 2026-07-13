import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_n_benthamiana.py"
SPEC = importlib.util.spec_from_file_location("preprocess_n_benthamiana", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.rules = MODULE.FilterRules(
            allowed_dna_alphabet=frozenset("ATCG"),
            required_start_codon="ATG",
            standard_stop_codons=frozenset(("TAA", "TAG", "TGA")),
            translation_table=1,
            max_protein_length=2045,
        )

    def test_valid_pair(self):
        self.assertEqual(MODULE.assess_pair("ATGAAATAA", "MK", self.rules), [])

    def test_terminal_stop_marker_is_representation_equivalent(self):
        self.assertEqual(MODULE.assess_pair("ATGAAATAA", "MK*", self.rules), [])

    def test_soft_masked_lowercase_is_evaluated_case_insensitively(self):
        self.assertEqual(MODULE.assess_pair("atgaaaTAA", "MK", self.rules), [])

    def test_all_rejection_reasons_are_preserved(self):
        reasons = MODULE.assess_pair("GTGNN", "M", self.rules)
        self.assertEqual(
            reasons,
            [
                "non_atcg",
                "cds_length_not_multiple_of_3",
                "translation_mismatch",
                "missing_atg_start",
                "missing_standard_stop",
            ],
        )

    def test_output_row_keeps_original_sequences(self):
        row = MODULE.make_row(
            "example", "ATGNNNTAA", "MX", "Nicotiana tabacum", "Nicotiana benthamiana"
        )
        self.assertEqual(row["dna"], "ATGNNNTAA")
        self.assertEqual(row["protein"], "MX")


if __name__ == "__main__":
    unittest.main()
