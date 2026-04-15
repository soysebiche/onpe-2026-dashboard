"""
m2_diff_in_diff.py

M2 — Diferencias-en-diferencias temporal 2016 vs 2026 a nivel distrito.

Pregunta: ¿Los distritos con mayor retraso en 2026 muestran un aumento
de ausentismo desde 2016 MAYOR que los distritos sin retraso?

Si β es significativo y positivo → el retraso logístico causó ausentismo
adicional más allá del aumento estructural observado.

También corre el PLACEBO TEST: ¿el retraso en 2026 predice el ausentismo
en 2016? (Debería ser cero si el retraso fue exógeno.)

Outputs:
  output/m2_resultados.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Cargar y preparar data 2016 (Lima, por distrito)
# ---------------------------------------------------------------------------

def load_2016_por_distrito() -> pd.DataFrame:
    df = pd.read_csv(
        ROOT / "output" / "eg2016_presidencial.csv",
        sep=";", encoding="latin-1"
    )
    # Solo Lima Metropolitana
    lima = df[df["DEPARTAMENTO"] == "LIMA"].copy()

    voto_cols = [c for c in lima.columns if c.startswith("VOTOS_")]
    lima["total_votos_2016"] = lima[voto_cols].sum(axis=1)
    lima = lima[lima["N_ELEC_HABIL"] > 0].copy()
    lima = lima[lima["total_votos_2016"] <= lima["N_ELEC_HABIL"]].copy()

    by_dist = lima.groupby("UBIGEO").agg(
        electores_2016=("N_ELEC_HABIL", "sum"),
        votantes_2016=("total_votos_2016", "sum"),
        mesas_2016=("MESA_DE_VOTACION", "count"),
        nombre_distrito=("DISTRITO", "first"),
    ).reset_index()

    by_dist["ausentismo_2016"] = (
        1 - by_dist["votantes_2016"] / by_dist["electores_2016"]
    ) * 100

    return by_dist


# ---------------------------------------------------------------------------
# 2. Cargar data 2026 por distrito (del CSV de distritos ya generado)
# ---------------------------------------------------------------------------

def load_2026_por_distrito() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "output" / "lima_instalacion_distritos.csv")
    return df.rename(columns={"districtUbigeo": "UBIGEO"})


# ---------------------------------------------------------------------------
# 3. Match y construcción del panel de distritos
# ---------------------------------------------------------------------------

def build_panel(d16: pd.DataFrame, d26: pd.DataFrame) -> pd.DataFrame:
    panel = d16.merge(d26, on="UBIGEO", how="inner")

    panel["delta_ausentismo"] = panel["ausentismoPctObservado"] - panel["ausentismo_2016"]
    panel["ausentismo_2026"] = panel["ausentismoPctObservado"]
    panel["pct_mesas_despues_8"] = panel["mesasDespues8"] / panel["mesasConHora"] * 100
    panel["pct_mesas_despues_10"] = panel["mesasDespues10"] / panel["mesasConHora"] * 100
    panel["delay_promedio"] = panel["delayPromedioHoras"]
    panel["mediana_hora"] = panel["horaMediana"]

    return panel.dropna(subset=["delta_ausentismo", "delay_promedio"])


# ---------------------------------------------------------------------------
# 4. Modelos
# ---------------------------------------------------------------------------

def run_ols(y, X_df, desc: str, weight_col=None, w=None) -> dict:
    X = sm.add_constant(X_df)
    if w is not None:
        model = sm.WLS(y, X, weights=w)
    else:
        model = sm.OLS(y, X)
    res = model.fit(cov_type="HC1")

    coefs = {}
    for i, name in enumerate(X.columns):
        coefs[name] = {
            "coef": float(res.params[i]),
            "se": float(res.bse[i]),
            "pvalue": float(res.pvalues[i]),
            "ci95_low": float(res.conf_int()[i][0]),
            "ci95_high": float(res.conf_int()[i][1]),
        }
    return {
        "descripcion": desc,
        "n": int(res.nobs),
        "r2": float(res.rsquared),
        "coeficientes": coefs,
    }


def run_all_models(panel: pd.DataFrame) -> dict:
    w = panel["electoresObservados"].values

    # -- Modelo M2a: DiD con variable continua (delay promedio)
    m2a = run_ols(
        panel["delta_ausentismo"],
        panel[["delay_promedio"]],
        "M2a — DiD: Δausentismo ~ delay_promedio (dist)",
        w=w,
    )

    # -- Modelo M2b: DiD con % mesas después de las 8am
    m2b = run_ols(
        panel["delta_ausentismo"],
        panel[["pct_mesas_despues_8"]],
        "M2b — DiD: Δausentismo ~ %mesas_despues_8 (dist)",
        w=w,
    )

    # -- Modelo M2c: DiD con % mesas después de las 10am
    m2c = run_ols(
        panel["delta_ausentismo"],
        panel[["pct_mesas_despues_10"]],
        "M2c — DiD: Δausentismo ~ %mesas_despues_10 (dist)",
        w=w,
    )

    # -- Placebo test A: ausentismo 2016 ~ delay 2026
    # Si el retraso fue exógeno, no debe predecir el ausentismo PRE-existente.
    placebo_a = run_ols(
        panel["ausentismo_2016"],
        panel[["delay_promedio"]],
        "Placebo A — ausentismo_2016 ~ delay_2026 (dist, continuo)",
        w=w,
    )

    # -- Placebo test B: ausentismo 2016 ~ %mesas_tarde 2026
    placebo_b = run_ols(
        panel["ausentismo_2016"],
        panel[["pct_mesas_despues_8"]],
        "Placebo B — ausentismo_2016 ~ %mesas_despues_8_2026 (dist)",
        w=w,
    )

    return {
        "M2a": m2a,
        "M2b": m2b,
        "M2c": m2c,
        "placebo_A": placebo_a,
        "placebo_B": placebo_b,
    }


# ---------------------------------------------------------------------------
# 5. Imputación DiD: cuánto del delta es atribuible al retraso
# ---------------------------------------------------------------------------

def imputation_did(panel: pd.DataFrame, m2a: dict) -> dict:
    """
    Usando el coeficiente de M2a, calcula el ausentismo extra atribuible al retraso
    en cada distrito y agrega a Lima.
    """
    beta = m2a["coeficientes"]["delay_promedio"]["coef"]
    panel = panel.copy()
    panel["ausentismo_extra_did"] = beta * panel["delay_promedio"]
    panel["ausentismo_extra_did"] = panel["ausentismo_extra_did"].clip(lower=0)
    panel["ausentes_extra_did"] = (panel["ausentismo_extra_did"] / 100) * panel["electoresObservados"]

    total_extra = float(panel["ausentes_extra_did"].sum())
    total_padron = float(panel["electoresObservados"].sum())
    total_delta = float((panel["delta_ausentismo"] * panel["electoresObservados"]).sum() / total_padron)

    return {
        "beta_delay": float(beta),
        "total_ausentes_extra_did": total_extra,
        "total_padron_lima": total_padron,
        "delta_ausentismo_promedio_lima": total_delta,
        "fraccion_delta_atribuible": total_extra / (total_delta / 100 * total_padron) if total_delta > 0 else None,
    }


# ---------------------------------------------------------------------------
# 6. Print y output
# ---------------------------------------------------------------------------

def print_results(models: dict, panel: pd.DataFrame, imp: dict) -> None:
    print("\n" + "=" * 72)
    print("M2 — DIFERENCIAS-EN-DIFERENCIAS 2016 → 2026 (nivel distrito)")
    print("=" * 72)

    print(f"\nDistritos en el panel: {len(panel)}")
    print(f"Ausentismo Lima 2016 (ponderado): {(panel['ausentismo_2016'] * panel['electores_2016']).sum() / panel['electores_2016'].sum():.2f}%")
    print(f"Ausentismo Lima 2026 (ponderado): {(panel['ausentismo_2026'] * panel['electoresObservados']).sum() / panel['electoresObservados'].sum():.2f}%")
    print(f"Δ ausentismo Lima (2026 − 2016):  {imp['delta_ausentismo_promedio_lima']:+.2f} pp")

    print("\n--- MODELOS DiD ---")
    for key in ["M2a", "M2b", "M2c"]:
        m = models[key]
        main_var = [k for k in m["coeficientes"] if k != "const"][0]
        c = m["coeficientes"][main_var]
        sig = "***" if c["pvalue"] < 0.01 else ("**" if c["pvalue"] < 0.05 else ("*" if c["pvalue"] < 0.1 else "n.s."))
        print(f"\n[{key}] {m['descripcion']}")
        print(f"  β = {c['coef']:+.4f}  SE={c['se']:.4f}  p={c['pvalue']:.3f} {sig}")
        print(f"  IC 95% = [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]")
        print(f"  R² = {m['r2']:.4f}  N = {m['n']}")

    print("\n--- PLACEBO TESTS (esperado: no significativo) ---")
    for key in ["placebo_A", "placebo_B"]:
        m = models[key]
        main_var = [k for k in m["coeficientes"] if k != "const"][0]
        c = m["coeficientes"][main_var]
        sig = "***" if c["pvalue"] < 0.01 else ("**" if c["pvalue"] < 0.05 else ("*" if c["pvalue"] < 0.1 else "n.s."))
        status = "PROBLEMA" if c["pvalue"] < 0.1 else "OK"
        print(f"\n[{key}] {m['descripcion']}")
        print(f"  β = {c['coef']:+.4f}  SE={c['se']:.4f}  p={c['pvalue']:.3f} {sig}  → {status}")

    print(f"\n--- IMPUTACIÓN DiD ---")
    print(f"  Δ total Lima: +{imp['delta_ausentismo_promedio_lima']:.2f} pp  ({int(imp['delta_ausentismo_promedio_lima']/100*imp['total_padron_lima']):,} ausentes extra vs 2016)")
    print(f"  Atribuible al retraso (DiD): ~{int(imp['total_ausentes_extra_did']):,} ausentes extra")
    if imp["fraccion_delta_atribuible"]:
        print(f"  Fracción del delta explicada por el retraso: {imp['fraccion_delta_atribuible']*100:.1f}%")
    print("=" * 72)


def main() -> None:
    print("[cargando data 2016...]")
    d16 = load_2016_por_distrito()
    print(f"  Distritos 2016 Lima: {len(d16)}")

    print("[cargando data 2026...]")
    d26 = load_2026_por_distrito()
    print(f"  Distritos 2026 Lima: {len(d26)}")

    print("[construyendo panel...]")
    panel = build_panel(d16, d26)
    print(f"  Distritos en el panel (match): {len(panel)}")

    print("[corriendo modelos...]")
    models = run_all_models(panel)

    imp = imputation_did(panel, models["M2a"])

    print_results(models, panel, imp)

    result = {
        "modelos": models,
        "imputacion_did": imp,
        "n_distritos": len(panel),
        "ausentismo_lima_2016_ponderado": float(
            (panel["ausentismo_2016"] * panel["electores_2016"]).sum()
            / panel["electores_2016"].sum()
        ),
        "ausentismo_lima_2026_ponderado": float(
            (panel["ausentismo_2026"] * panel["electoresObservados"]).sum()
            / panel["electoresObservados"].sum()
        ),
    }

    out = ROOT / "output" / "m2_resultados.json"
    with out.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[ok] guardado en {out}")


if __name__ == "__main__":
    main()
