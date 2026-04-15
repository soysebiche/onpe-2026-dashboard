"""
Calibración del modelo bayesiano usando EG2021 como ground truth.

Estrategia:
  - Los distritos de mayor altitud tienden a reportar tarde (rurales, dispersos).
  - Simulamos "snapshots" al 50%, 60%, 70%, 80%, 90% contando primero los
    distritos de menor altitud (urbanos) y dejando los de mayor altitud para el final.
  - En cada snapshot corremos el modelo Dirichlet-Multinomial y comparamos la
    proyección vs el resultado real final.
  - Medimos:
      · Error sistemático (sesgo): proyección - real, por candidato.
      · Cobertura: ¿el resultado real cayó dentro del IC 95%?
      · Swing óptimo: el mínimo que logra ≥90% de cobertura.

Uso:
    python3 calibrar_2021.py
    python3 calibrar_2021.py --swing-range 0.01 0.02 0.03 0.05 0.07 0.10
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("/tmp/presidencial-resultados-partidos.csv")
RENIEC_CSV = Path("/tmp/ubigeo_reniec.csv")
UMBRAL_RURAL = 15_000


# ── helpers ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").upper().strip()


def cargar_2021() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)
    # ubigeo ONPE nivel3 = int(ubigeo[:6])
    df["ubigeo3_onpe"] = df["ubigeo"].apply(lambda x: int(x[:2]) * 10000
                                              + int(x[2:4]) * 100
                                              + int(x[4:6]))
    df["ubigeo1_onpe"] = df["ubigeo"].apply(lambda x: int(x[:2]) * 10000)
    df["altitud"] = pd.to_numeric(df["altitud"], errors="coerce").fillna(0)
    return df


def cargar_poblacion() -> dict[int, int]:
    """RENIEC: ubigeo3_onpe → población"""
    if not RENIEC_CSV.exists():
        return {}
    pob = {}
    with open(RENIEC_CSV, encoding="utf-8-sig") as f:
        import csv
        for row in csv.DictReader(f):
            raw = row.get("Ubigeo", "").strip().zfill(6)
            if len(raw) < 6:
                continue
            key = int(raw[:2]) * 10000 + int(raw[2:4]) * 100 + int(raw[4:6])
            try:
                pob[key] = int(row.get("Poblacion", "0").replace(",", ""))
            except ValueError:
                pob[key] = 0
    return pob


# ── simulación de snapshot ───────────────────────────────────────────────────

def simular_snapshot(
    df: pd.DataFrame,
    pob: dict[int, int],
    pct_objetivo: float,
    orden: str = "altitud",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (df_contado, df_faltante) simulando que se ha contado `pct_objetivo`
    de los votos totales. Los distritos se ordenan por `orden` (asc) — los primeros
    se "cuentan" antes, los últimos llegan tarde.

    orden='altitud'   → urbanos primero (baja altitud), rurales tarde
    orden='poblacion' → grandes primero, chicos tarde
    """
    dist_stats = df.groupby("ubigeo3_onpe").agg(
        total_votos=("total_votos", "sum"),
        altitud=("altitud", "first"),
        ubigeo1=("ubigeo1_onpe", "first"),
    ).reset_index()

    if orden == "altitud":
        dist_stats = dist_stats.sort_values("altitud", ascending=True)
    elif orden == "poblacion":
        dist_stats["pob"] = dist_stats["ubigeo3_onpe"].map(pob).fillna(0)
        dist_stats = dist_stats.sort_values("pob", ascending=False)

    votos_total = dist_stats["total_votos"].sum()
    objetivo = votos_total * pct_objetivo / 100.0

    acum = 0
    contados = []
    for _, row in dist_stats.iterrows():
        if acum >= objetivo:
            break
        contados.append(row["ubigeo3_onpe"])
        acum += row["total_votos"]

    df_cont = df[df["ubigeo3_onpe"].isin(set(contados))]
    df_falt = df[~df["ubigeo3_onpe"].isin(set(contados))]
    return df_cont, df_falt


# ── modelo (simplificado, sin API) ───────────────────────────────────────────

def proyectar_2021(
    df_cont: pd.DataFrame,
    df_falt: pd.DataFrame,
    swing: float = 0.03,
    draws: int = 5_000,
    rng=None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Corre el modelo Dirichlet-Multinomial sobre df_cont y proyecta df_falt.
    Devuelve (candidatos, mediana_pct, ic95 array (K,2)).
    """
    rng = rng or np.random.default_rng(42)
    concentracion = 1.0 / swing**2 if swing > 0 else float("inf")

    candidatos = sorted(df_cont["candidato"].unique())
    K = len(candidatos)
    idx = {c: i for i, c in enumerate(candidatos)}

    # Agregar votos contados por departamento (prior rural)
    dept_agg: dict[int, np.ndarray] = {}
    for (u1, cand), grp in df_cont.groupby(["ubigeo1_onpe", "candidato"]):
        if u1 not in dept_agg:
            dept_agg[u1] = np.zeros(K, dtype=np.int64)
        dept_agg[u1][idx[cand]] += int(grp["total_votos"].sum())
    national_agg = sum(dept_agg.values(), np.zeros(K, dtype=np.int64))

    contado_total = np.zeros(K, dtype=np.int64)
    sims = np.zeros((draws, K), dtype=np.int64)

    # Por distrito contado: acumular votos reales
    dist_votos_cont: dict[int, np.ndarray] = {}
    for (u3, cand), grp in df_cont.groupby(["ubigeo3_onpe", "candidato"]):
        if u3 not in dist_votos_cont:
            dist_votos_cont[u3] = np.zeros(K, dtype=np.int64)
        dist_votos_cont[u3][idx[cand]] += int(grp["total_votos"].sum())
    for v in dist_votos_cont.values():
        contado_total += v

    # Por distrito faltante: proyectar usando prior departamental
    dist_votos_falt: dict[int, tuple[int, np.ndarray]] = {}
    for (u3, cand), grp in df_falt.groupby(["ubigeo3_onpe", "candidato"]):
        v_total = int(grp["total_votos"].sum())
        if u3 not in dist_votos_falt:
            u1 = int(df_falt[df_falt["ubigeo3_onpe"] == u3]["ubigeo1_onpe"].iloc[0])
            dist_votos_falt[u3] = (u1, np.zeros(K, dtype=np.int64))
        dist_votos_falt[u3][1][idx[cand]] = v_total

    for u3, (u1, votos_reales) in dist_votos_falt.items():
        v_falt = int(votos_reales.sum())
        if v_falt <= 0:
            continue

        # Prior = departamental rural
        ref = dept_agg.get(u1, national_agg)
        ref_sum = float(ref.sum())
        if ref_sum > 0 and np.isfinite(concentracion):
            alpha = 1.0 + (ref / ref_sum) * concentracion
        else:
            alpha = np.full(K, 1.0 / K)

        p_draws = rng.dirichlet(alpha, size=draws)
        for i in range(draws):
            sims[i] += rng.multinomial(v_falt, p_draws[i])

    proyectado = sims + contado_total
    pct = proyectado / proyectado.sum(axis=1, keepdims=True) * 100.0

    med = np.median(pct, axis=0)
    lo = np.percentile(pct, 2.5, axis=0)
    hi = np.percentile(pct, 97.5, axis=0)
    ic = np.stack([lo, hi], axis=1)
    return candidatos, med, ic


# ── calibración ──────────────────────────────────────────────────────────────

def calibrar(swing_vals: list[float], snapshots: list[float], draws: int = 3_000):
    print("Cargando datos 2021…")
    df = cargar_2021()
    pob = cargar_poblacion()

    # Resultado real final
    real = df.groupby("candidato")["total_votos"].sum()
    total_real = real.sum()
    real_pct = (real / total_real * 100).to_dict()

    print(f"Total votos 2021: {total_real:,} | {len(real)} candidatos\n")

    # Para cada snapshot y cada swing: medir sesgo y cobertura
    resultados = []

    for snap in snapshots:
        print(f"━━ Snapshot {snap:.0f}% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        df_cont, df_falt = simular_snapshot(df, pob, snap, orden="altitud")
        votos_cont = df_cont["total_votos"].sum()
        print(f"   Votos contados: {votos_cont:,} ({votos_cont/total_real*100:.1f}%)")
        print(f"   Distritos contados: {df_cont['ubigeo3_onpe'].nunique()} / "
              f"{df['ubigeo3_onpe'].nunique()}")

        for swing in swing_vals:
            rng = np.random.default_rng(42)
            cands, med, ic = proyectar_2021(df_cont, df_falt, swing=swing,
                                             draws=draws, rng=rng)

            errores = {}
            coberturas = {}
            for i, c in enumerate(cands):
                r = real_pct.get(c, 0.0)
                err = med[i] - r  # positivo = sobreestima, negativo = subestima
                dentro = ic[i, 0] <= r <= ic[i, 1]
                errores[c] = err
                coberturas[c] = dentro

            cobertura_global = np.mean(list(coberturas.values()))
            rmse = np.sqrt(np.mean([e**2 for e in errores.values()]))

            resultados.append({
                "snapshot": snap,
                "swing": swing,
                "cobertura": cobertura_global,
                "rmse": rmse,
                "errores": errores,
            })

            print(f"   swing={swing:.2f}  cobertura={cobertura_global*100:.0f}%  "
                  f"RMSE={rmse:.3f}pp")

        print()

    # ── Reporte de sesgo sistemático ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("SESGO SISTEMÁTICO por candidato (proyección - real)")
    print("Snapshot 70%, swing=0.03 (configuración actual del modelo)")
    print("=" * 70)

    res_70 = next((r for r in resultados if r["snapshot"] == 70 and r["swing"] == 0.03), None)
    if res_70:
        errores = res_70["errores"]
        print(f"\n{'Candidato':<42} {'Error (pp)':>12}  {'Dirección'}")
        print("-" * 70)
        for c, err in sorted(errores.items(), key=lambda x: -abs(x[1])):
            dir_str = "↑ SOBREESTIMA" if err > 0.3 else ("↓ SUBESTIMA" if err < -0.3 else "≈ ok")
            print(f"  {c[:40]:<40} {err:>+10.2f}pp  {dir_str}")

    # ── Cobertura por swing a 70% ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COBERTURA IC95% por valor de swing (snapshot 70%)")
    print("(objetivo: ≥90% de candidatos dentro del IC)")
    print("=" * 70)
    for r in resultados:
        if r["snapshot"] == 70:
            marca = " ← RECOMENDADO" if r["cobertura"] >= 0.90 and \
                    all(rr["cobertura"] < 0.90 for rr in resultados
                        if rr["snapshot"] == 70 and rr["swing"] < r["swing"]) else ""
            print(f"  swing={r['swing']:.2f}  cobertura={r['cobertura']*100:.0f}%  "
                  f"RMSE={r['rmse']:.3f}pp{marca}")

    # ── Swing óptimo ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SWING ÓPTIMO por snapshot")
    print("=" * 70)
    for snap in snapshots:
        validos = [r for r in resultados if r["snapshot"] == snap and r["cobertura"] >= 0.90]
        if validos:
            mejor = min(validos, key=lambda x: x["rmse"])
            print(f"  {snap:.0f}% → swing={mejor['swing']:.2f}  "
                  f"cobertura={mejor['cobertura']*100:.0f}%  RMSE={mejor['rmse']:.3f}pp")
        else:
            peor = max([r for r in resultados if r["snapshot"] == snap],
                       key=lambda x: x["cobertura"])
            print(f"  {snap:.0f}% → ningún swing logra ≥90%  "
                  f"(mejor: swing={peor['swing']:.2f}, {peor['cobertura']*100:.0f}%)")

    return resultados


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swing-range", type=float, nargs="+",
                    default=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10])
    ap.add_argument("--snapshots", type=float, nargs="+",
                    default=[60, 70, 80, 90])
    ap.add_argument("--draws", type=int, default=3_000)
    args = ap.parse_args()

    calibrar(args.swing_range, args.snapshots, draws=args.draws)


if __name__ == "__main__":
    main()
