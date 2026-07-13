import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_cluster_splits import assign_clusters, merged_codon_tokens
from scripts.cluster_proteins import parse_mmseqs_clusters
from scripts.compute_codon_metrics import (
    genetic_code_families,
    geometric_codon_score,
    quantile,
    relative_adaptiveness,
)
from scripts.validate_cluster_splits import validate_tokens


class CodonMetricTests(unittest.TestCase):
    def test_relative_adaptiveness_and_geometric_score(self):
        families, sense_codons, _ = genetic_code_families(1)
        counts = Counter({codon: 0 for codon in sense_codons})
        counts.update({"AAA": 9, "AAG": 1})
        weights = relative_adaptiveness(counts, families, pseudocount=0)
        self.assertEqual(weights["AAA"], 1.0)
        self.assertAlmostEqual(weights["AAG"], 1 / 9)
        score = geometric_codon_score(["AAA", "AAG"], weights, families)
        self.assertAlmostEqual(score, 1 / 3)

    def test_quantile_uses_linear_interpolation(self):
        self.assertEqual(quantile([0.0, 1.0], 0.25), 0.25)
        self.assertTrue(math.isclose(quantile([1, 2, 3, 4, 5], 0.5), 3))


class ClusterTests(unittest.TestCase):
    def test_mmseqs_membership_is_complete_and_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.tsv"
            path.write_text("a\ta\na\tb\nc\tc\n", encoding="utf-8")
            rows, clusters = parse_mmseqs_clusters(path, {"a", "b", "c"})
        self.assertEqual({row["source_id"] for row in rows}, {"a", "b", "c"})
        self.assertEqual(sorted(map(len, clusters.values())), [1, 2])

    def test_cluster_assignment_is_deterministic_and_atomic(self):
        sizes = {"a": 8, "b": 4, "c": 3, "d": 2, "e": 1, "f": 1, "g": 1}
        ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
        first, counts = assign_clusters(sizes, ratios, seed=23)
        second, _ = assign_clusters(sizes, ratios, seed=23)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(sizes))
        self.assertEqual(sum(counts.values()), sum(sizes.values()))

    def test_cluster_assignment_does_not_systematically_isolate_large_clusters(self):
        sizes = {f"cluster_{index:03d}": (index % 5) + 1 for index in range(100)}
        ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
        assignments, _ = assign_clusters(sizes, ratios, seed=23)
        cluster_counts = Counter(assignments.values())
        self.assertGreater(cluster_counts["validation"], 5)
        self.assertGreater(cluster_counts["test"], 5)

    def test_upstream_token_format(self):
        self.assertEqual(
            merged_codon_tokens("MK", "ATGAAGTAA"),
            "M_ATG K_AAG __TAA",
        )

    def test_exported_tokens_translate(self):
        self.assertEqual(validate_tokens("M_ATG K_AAG __TAA"), 3)
        with self.assertRaises(ValueError):
            validate_tokens("M_ATG E_AAA __TAA")


if __name__ == "__main__":
    unittest.main()
