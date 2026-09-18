# pattern: Functional Core
import polars as pl

from scripts.plot_component_proportions import _build_comparison_dataframe


def test_comparison_preserves_noncontiguous_truth_component_ids():
    truth = pl.DataFrame(
        {
            "component": [0, 0, 2, 2],
            "sigma2": [0.01, 0.01, 1.0, 1.0],
            "effect_allele_frequency": [0.1, 0.2, 0.1, 0.2],
        }
    )
    inferred = pl.DataFrame({"maf": [0.01, 0.01], "var0": [0.01, 1.0], "value": [0.5, 0.5]})
    comparison, skipped = _build_comparison_dataframe(truth, inferred, maf_min=0.0)
    assert skipped == []
    assert comparison["component"].to_list() == [0, 2]
    assert comparison["simulated_proportion"].to_list() == [0.5, 0.5]
    assert comparison["inferred_proportion"].to_list() == [0.5, 0.5]
    assert comparison["true_component_sigma2"].to_list() == [0.01, 1.0]
