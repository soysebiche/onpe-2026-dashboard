"""
M1 — Estimación del efecto del retraso en instalación sobre el ausentismo,
con efectos fijos de local de votación.

Compara:
  A) OLS ponderado simple (reproducción del modelo ingenuo del pipeline).
  B) M1: OLS con efectos fijos de local (dentro-local), ponderado, SE clusterizados.
  C) M3: Efecto no lineal con bins de retraso + efectos fijos de local.

Input:  output/lima_instalacion_mesas.csv
Output: output/m1_resultados.json  +  tabla impresa a stdout
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

CSV_PATH = Path("output/lima_instalacion_mesas.csv")
OUT_PATH = Path("output/m1_resultados.json")


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    n0 = len(df)

    # Solo actas contabilizadas y con hora de instalación extraída.
    df = df[df["codigoEstadoActa"] == "C"]
    df = df[df["installHour"].notna()]
    df = df[df["delayHours"].notna()]
    df = df[df["totalElectoresHabiles"] > 0]
    df = df[df["ausentismoPct"].notna()]
    df = df[df["codigoLocalVotacion"].notna()]

    # Filtros de sanidad: hora entre 04:00 y 16:00.
    df = df[(df["installHour"] >= 4) & (df["installHour"] <= 16)]

    # Ausentismo en [0, 100].
    df = df[(df["ausentismoPct"] >= 0) & (df["ausentismoPct"] <= 100)]

    print(f"[load] filas iniciales: {n0}")
    print(f"[load] filas después de filtros: {len(df)}")
    print(f"[load] locales únicos: {df['codigoLocalVotacion'].nunique()}")
    print(f"[load] mesas con >=2 mesas en el mismo local: "
          f"{df.groupby('codigoLocalVotacion').size().gt(1).sum()} locales")
    return df.reset_index(drop=True)


def weighted_ols(y: np.ndarray, X: np.ndarray, w: np.ndarray,
                 cluster_ids: np.ndarray | None = None) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS ponderado (WLS) con opción de SE clusterizados."""
    model = sm.WLS(y, X, weights=w)
    if cluster_ids is not None:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_ids})
    else:
        res = model.fit(cov_type="HC1")
    return res


def within_transform(df: pd.DataFrame, cols: list[str], group_col: str,
                      weight_col: str) -> pd.DataFrame:
    """Transformación within (demeaning) ponderada por local de votación."""
    out = df[[group_col, weight_col] + cols].copy()
    w = out[weight_col]
    for c in cols:
        # Media ponderada por local.
        grouped = out.groupby(group_col)
        wsum = grouped[weight_col].transform("sum")
        wy = out[c] * w
        wymean = grouped[wy.name if hasattr(wy, "name") else c].transform(
            lambda s: s.sum()
        ) if False else None  # placeholder
    # El enfoque anterior es innecesariamente complejo; lo hacemos manual.
    result = df[[group_col, weight_col]].copy()
    grp = df.groupby(group_col)
    wsum = grp[weight_col].transform("sum")
    for c in cols:
        numer = (df[c] * df[weight_col]).groupby(df[group_col]).transform("sum")
        wmean = numer / wsum
        result[c] = df[c] - wmean
    return result


def model_a_naive(df: pd.DataFrame) -> dict:
    """OLS ponderado ingenuo: ausentismo ~ delay, sin FE."""
    y = df["ausentismoPct"].to_numpy()
    X = sm.add_constant(df["delayHours"].to_numpy())
    w = df["totalElectoresHabiles"].to_numpy()
    res = weighted_ols(y, X, w)
    return {
        "modelo": "A — OLS ponderado (sin efectos fijos)",
        "n": int(res.nobs),
        "intercept": float(res.params[0]),
        "slope_pp_per_hour": float(res.params[1]),
        "slope_se": float(res.bse[1]),
        "slope_ci95_low": float(res.conf_int()[1][0]),
        "slope_ci95_high": float(res.conf_int()[1][1]),
        "r2": float(res.rsquared),
    }


def model_b_fixed_effects(df: pd.DataFrame) -> dict:
    """M1 — OLS con efectos fijos de local (within transformation)."""
    # Dropear locales singleton (una sola mesa) — no aportan a within.
    grp_sizes = df.groupby("codigoLocalVotacion").size()
    multi_locals = grp_sizes[grp_sizes >= 2].index
    d = df[df["codigoLocalVotacion"].isin(multi_locals)].copy()

    demeaned = within_transform(
        d,
        cols=["ausentismoPct", "delayHours"],
        group_col="codigoLocalVotacion",
        weight_col="totalElectoresHabiles",
    )

    y = demeaned["ausentismoPct"].to_numpy()
    X = demeaned["delayHours"].to_numpy().reshape(-1, 1)
    w = d["totalElectoresHabiles"].to_numpy()
    cluster = d["codigoLocalVotacion"].to_numpy()

    # Sin constante: ya está absorbida por la demeaning.
    res = weighted_ols(y, X, w, cluster_ids=cluster)

    # R² within: varianza explicada sobre la varianza within.
    y_pred = res.predict(X)
    ss_res = float(np.sum(w * (y - y_pred) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2_within = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "modelo": "B — M1: Efectos fijos de local de votación",
        "n": int(res.nobs),
        "n_locales": int(d["codigoLocalVotacion"].nunique()),
        "slope_pp_per_hour": float(res.params[0]),
        "slope_se_clustered": float(res.bse[0]),
        "slope_ci95_low": float(res.conf_int()[0][0]),
        "slope_ci95_high": float(res.conf_int()[0][1]),
        "r2_within": float(r2_within),
        "nota": "SE clusterizados por local de votación. Singletons dropeados.",
    }


def model_c_nonlinear_fe(df: pd.DataFrame) -> dict:
    """M3 — Efecto no lineal con bins de retraso + efectos fijos de local."""
    grp_sizes = df.groupby("codigoLocalVotacion").size()
    multi_locals = grp_sizes[grp_sizes >= 2].index
    d = df[df["codigoLocalVotacion"].isin(multi_locals)].copy()

    # Bins de retraso (en horas). Referencia: retraso <= 0.5h (apertura "normal").
    bins = [-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf]
    labels = ["bin_ref_0_05", "bin_05_1", "bin_1_2", "bin_2_3", "bin_3plus"]
    d["delay_bin"] = pd.cut(d["delayHours"], bins=bins, labels=labels)

    dummies = pd.get_dummies(d["delay_bin"], drop_first=True).astype(float)

    # Within transform para las dummies y la y.
    work = pd.concat([
        d[["codigoLocalVotacion", "totalElectoresHabiles", "ausentismoPct"]],
        dummies,
    ], axis=1)
    demeaned_cols = ["ausentismoPct"] + list(dummies.columns)
    demeaned = within_transform(
        work,
        cols=demeaned_cols,
        group_col="codigoLocalVotacion",
        weight_col="totalElectoresHabiles",
    )

    y = demeaned["ausentismoPct"].to_numpy()
    X = demeaned[list(dummies.columns)].to_numpy()
    w = d["totalElectoresHabiles"].to_numpy()
    cluster = d["codigoLocalVotacion"].to_numpy()

    res = weighted_ols(y, X, w, cluster_ids=cluster)

    coefs = {}
    for i, name in enumerate(dummies.columns):
        coefs[name] = {
            "coef_pp": float(res.params[i]),
            "se": float(res.bse[i]),
            "ci95_low": float(res.conf_int()[i][0]),
            "ci95_high": float(res.conf_int()[i][1]),
            "n_mesas_en_bin": int((d["delay_bin"] == name).sum()),
        }

    return {
        "modelo": "C — M3: Efecto no lineal (bins) + efectos fijos de local",
        "n": int(res.nobs),
        "bin_ref": "retraso ≤ 0.5h (mesas 'a tiempo')",
        "coeficientes_por_bin": coefs,
        "nota": "Cada coeficiente es el efecto en pp de ausentismo vs. el bin de referencia, manteniendo local constante.",
    }


def print_comparison(results: dict) -> None:
    print("\n" + "=" * 72)
    print("COMPARACIÓN DE MODELOS — Efecto del retraso sobre el ausentismo")
    print("=" * 72)

    a = results["modelo_A_naive"]
    b = results["modelo_B_fixed_effects"]

    print(f"\n[A] {a['modelo']}")
    print(f"    N = {a['n']}")
    print(f"    β = {a['slope_pp_per_hour']:.4f} pp/hora")
    print(f"    SE = {a['slope_se']:.4f}")
    print(f"    IC 95% = [{a['slope_ci95_low']:.4f}, {a['slope_ci95_high']:.4f}]")
    print(f"    R² = {a['r2']:.5f}")

    print(f"\n[B] {b['modelo']}")
    print(f"    N = {b['n']}   (locales = {b['n_locales']})")
    print(f"    β = {b['slope_pp_per_hour']:.4f} pp/hora")
    print(f"    SE clusterizado = {b['slope_se_clustered']:.4f}")
    print(f"    IC 95% = [{b['slope_ci95_low']:.4f}, {b['slope_ci95_high']:.4f}]")
    print(f"    R² within = {b['r2_within']:.5f}")

    delta = b["slope_pp_per_hour"] - a["slope_pp_per_hour"]
    pct = (delta / a["slope_pp_per_hour"]) * 100 if a["slope_pp_per_hour"] != 0 else 0
    print(f"\n    Δβ (B - A) = {delta:+.4f} pp/hora ({pct:+.1f}%)")

    c = results["modelo_C_nonlinear_fe"]
    print(f"\n[C] {c['modelo']}")
    print(f"    N = {c['n']}")
    print(f"    Referencia: {c['bin_ref']}")
    print(f"    {'Bin':<18} {'β (pp)':>10} {'SE':>10} {'IC95%':>22} {'N mesas':>10}")
    for name, info in c["coeficientes_por_bin"].items():
        ci = f"[{info['ci95_low']:+.2f}, {info['ci95_high']:+.2f}]"
        print(f"    {name:<18} {info['coef_pp']:>+10.3f} {info['se']:>10.3f} {ci:>22} {info['n_mesas_en_bin']:>10}")

    print("\n" + "=" * 72)


def main() -> None:
    df = load_and_clean()

    print("\n[modelo A] OLS ponderado ingenuo...")
    a = model_a_naive(df)

    print("[modelo B] M1 con efectos fijos de local...")
    b = model_b_fixed_effects(df)

    print("[modelo C] M3 no lineal con bins + FE...")
    c = model_c_nonlinear_fe(df)

    results = {
        "modelo_A_naive": a,
        "modelo_B_fixed_effects": b,
        "modelo_C_nonlinear_fe": c,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_comparison(results)
    print(f"\n[ok] Resultados guardados en {OUT_PATH}")


if __name__ == "__main__":
    main()
