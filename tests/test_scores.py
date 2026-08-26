"""Unit tests for `stem_dt.scores`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stem_dt import scores


def test_marker_mean_drops_missing_columns() -> None:
    expr = pd.DataFrame({"TH": [1.0, 2.0], "DDC": [3.0, 4.0]})
    out = scores.dopaminergic_score(expr, ["TH", "DDC", "MISSING"])
    np.testing.assert_allclose(out.values, [2.0, 3.0])


def test_marker_mean_all_missing_returns_nan() -> None:
    expr = pd.DataFrame({"FOO": [1.0]})
    out = scores.proteomics_score(expr, ["BAR", "BAZ"])
    assert out.isna().all()


def test_minmax_bounds_clip_to_unit_interval() -> None:
    train = np.array([[[0.0, 1.0], [0.5, 2.0]]], dtype=np.float32)  # (1,2,2)
    bounds = scores.fit_minmax(train)
    np.testing.assert_allclose(bounds.lo, [0.0, 1.0])
    np.testing.assert_allclose(bounds.hi, [0.5, 2.0])

    # Out-of-train values are clipped, not extrapolated (§1.4 Step 4).
    test = np.array([[-1.0, 5.0], [0.25, 1.5]])
    scaled = bounds.apply(test)
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0
    np.testing.assert_allclose(scaled[1], [0.5, 0.5])


def test_minmax_handles_zero_range() -> None:
    """If a dim has zero variance on train, scaling must not divide by zero."""
    train = np.array([[[0.7, 0.0], [0.7, 0.0]]], dtype=np.float32)
    bounds = scores.fit_minmax(train)
    out = bounds.apply(np.array([[0.7, 0.0], [0.7, 1.0]]))
    assert np.isfinite(out).all()


def test_split_trajectories_is_deterministic_and_partitions() -> None:
    tr, va, te = scores.split_trajectories(n=200, splits=(140, 30, 30), seed=42)
    assert len(tr) == 140 and len(va) == 30 and len(te) == 30
    assert set(tr).isdisjoint(va)
    assert set(tr).isdisjoint(te)
    assert set(va).isdisjoint(te)
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(200))

    # Determinism: same seed → identical indices.
    tr2, va2, te2 = scores.split_trajectories(n=200, splits=(140, 30, 30), seed=42)
    np.testing.assert_array_equal(tr, tr2)
    np.testing.assert_array_equal(va, va2)
    np.testing.assert_array_equal(te, te2)


def test_assemble_trajectories_shape_and_seed() -> None:
    df = pd.DataFrame(np.random.RandomState(0).rand(50, 4))  # 50 cells, 4 dims
    by_stage = {"D11": df, "D30": df, "D52": df}
    t1, order = scores.assemble_trajectories(by_stage, n_trajectories=10, seed=42)
    t2, _ = scores.assemble_trajectories(by_stage, n_trajectories=10, seed=42)
    assert t1.shape == (10, 3, 4)
    assert order == ["D11", "D30", "D52"]
    np.testing.assert_array_equal(t1, t2)


def test_phospho_score_uses_gene_residue_keys() -> None:
    log_phospho = pd.DataFrame(
        {"TH_S40": [0.5, 1.5], "MAPK1_T185": [1.0, 2.0]}
    )
    out = scores.phospho_score(log_phospho, [("TH", "S40"), ("MAPK1", "T185")])
    np.testing.assert_allclose(out.values, [0.75, 1.75])


def test_assign_stage_by_source_returns_label_or_nan() -> None:
    from stem_dt import harmonize

    samples = pd.DataFrame(
        {
            "source": ["Virdi2022", "Virdi2022", "Brenes2018", "Other"],
            "sample_label": ["D25", "D65", "hiPSC", "foo"],
        }
    )
    mapping = {
        ("Virdi2022", "D25"): "D30",
        ("Virdi2022", "D65"): "D52",
        ("Brenes2018", "hiPSC"): "D11",
    }
    out = harmonize.assign_stage_by_source(samples, mapping=mapping)
    assert out.iloc[:3].tolist() == ["D30", "D52", "D11"]
    assert pd.isna(out.iloc[3])


def test_stage_conditional_prior_is_constant_across_stages() -> None:
    from stem_dt import harmonize

    prot = pd.DataFrame({"TH": [1.0, 2.0, 3.0], "NR4A2": [0.5, 0.5, 1.5]})
    prior = harmonize.as_stage_conditional_prior(prot)
    assert prior.shape == (3, 2)
    np.testing.assert_allclose(prior.values, np.tile([2.0, 5 / 6], (3, 1)))


def test_synthetic_tensor_shape_and_signal() -> None:
    from stem_dt import synthetic

    tensor, stages = synthetic.generate_synthetic_state_tensor(n=200, seed=42)
    assert tensor.shape == (200, 3, 4)
    assert stages == ["D11", "D30", "D52"]
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    # P decreases, D increases monotonically across stages (means).
    means = tensor.mean(axis=0)
    assert means[0, 0] > means[1, 0] > means[2, 0]
    assert means[0, 1] < means[1, 1] < means[2, 1]


def test_synthetic_null_kills_pr_ph_signal() -> None:
    """M-F sanity: pr_signal=0 should make the Pr dim stage-independent."""
    from stem_dt import synthetic

    tensor, _ = synthetic.generate_synthetic_state_tensor(
        n=500, pr_signal=0.0, ph_signal=0.0, seed=42
    )
    means = tensor.mean(axis=0)
    pr_range = float(np.ptp(means[:, 2]))  # max - min across stages
    ph_range = float(np.ptp(means[:, 3]))
    assert pr_range < 0.05, f"Pr should be flat under null, got range={pr_range}"
    assert ph_range < 0.05, f"Ph should be flat under null, got range={ph_range}"


def test_permute_dims_breaks_per_cell_association_only() -> None:
    """Per-stage marginals are preserved; per-cell associations are not."""
    from stem_dt import synthetic

    tensor, _ = synthetic.generate_synthetic_state_tensor(n=200, seed=0)
    permuted = synthetic.permute_dims(tensor, dims_to_permute=(2, 3), seed=1)

    # Marginal mean per stage per dim is preserved (up to float32 summation order).
    np.testing.assert_allclose(
        tensor.mean(axis=0), permuted.mean(axis=0), rtol=0, atol=1e-6
    )

    # But Pr dim is no longer aligned to its original cells.
    assert not np.allclose(tensor[:, :, 2], permuted[:, :, 2])


def test_cross_modal_correlations_shapes_and_diagonal() -> None:
    from stem_dt import synthetic, validation

    tensor, stages = synthetic.generate_synthetic_state_tensor(n=200, seed=42)
    res = validation.cross_modal_correlations(
        tensor, stages, method="spearman", n_bootstrap=50, seed=42
    )
    assert set(res) == {"D11", "D30", "D52"}
    for stage in stages:
        rho = res[stage]["rho"]
        assert rho.shape == (4, 4)
        # Spearman self-correlation: scipy yields ~1 with floating precision.
        np.testing.assert_allclose(np.diag(rho), 1.0, atol=1e-9)
        # CIs should bracket the point estimate.
        assert (res[stage]["ci_low"] <= rho + 1e-9).all()
        assert (res[stage]["ci_high"] >= rho - 1e-9).all()


def test_cross_modal_correlations_under_null() -> None:
    """M-F sanity: with no Pr/Ph signal, correlations to P and D should be ~0."""
    from stem_dt import synthetic, validation

    tensor, stages = synthetic.generate_synthetic_state_tensor(
        n=500, pr_signal=0.0, ph_signal=0.0, seed=42
    )
    res = validation.cross_modal_correlations(
        tensor, stages, method="spearman", n_bootstrap=0
    )
    # |rho(Pr, D)| and |rho(Ph, D)| should be small under the null.
    for stage in stages:
        rho = res[stage]["rho"]
        assert abs(rho[1, 2]) < 0.2, f"{stage} Pr-D rho={rho[1, 2]:.3f}"
        assert abs(rho[1, 3]) < 0.2, f"{stage} Ph-D rho={rho[1, 3]:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
