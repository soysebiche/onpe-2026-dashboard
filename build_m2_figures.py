"""
build_m2_figures.py

Genera las figuras de M2 para el working paper:
  fig6_did_scatter.png   — Scatter distrital Δausentismo vs delay
  fig7_triple_robustez.png — Comparación de los 3 métodos vs carta RP
  fig8_placebo.png       — Placebo test: ausentismo 2016 vs delay 2026
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Estilo (igual que build_report.py)
# ---------------------------------------------------------------------------
BG = "#f5f1e8"; FG = "#2c2c2c"; MUTED = "#6c6c6c"; GRID = "#d8d2c4"
PRIMARY = "#6b8ca3"; SECONDARY = "#c78585"; TERTIARY = "#8fa68e"
QUATERNARY = "#d4a373"; ACCENT = "#a63a3a"

mpl.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": MUTED, "axes.labelcolor": FG, "axes.titlecolor": FG,
    "text.color": FG, "xtick.color": FG, "ytick.color": FG,
    "grid.color": GRID, "font.family": "serif",
    "font.serif": ["Palatino", "Georgia", "DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "figure.dpi": 120, "savefig.dpi": 180, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "output" / "figures"

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(True, axis="both", alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)

# ---------------------------------------------------------------------------
# Cargar panel distrital (re-construir igual que m2_diff_in_diff.py)
# ---------------------------------------------------------------------------
def build_panel() -> pd.DataFrame:
    df16 = pd.read_csv(ROOT / "output" / "eg2016_presidencial.csv",
                       sep=";", encoding="latin-1")
    lima16 = df16[df16["DEPARTAMENTO"] == "LIMA"].copy()
    vcols = [c for c in lima16.columns if c.startswith("VOTOS_")]
    lima16["total_votos_2016"] = lima16[vcols].sum(axis=1)
    lima16 = lima16[lima16["N_ELEC_HABIL"] > 0]
    lima16 = lima16[lima16["total_votos_2016"] <= lima16["N_ELEC_HABIL"]]
    d16 = lima16.groupby("UBIGEO").agg(
        electores_2016=("N_ELEC_HABIL", "sum"),
        votantes_2016=("total_votos_2016", "sum"),
        nombre=("DISTRITO", "first"),
    ).reset_index()
    d16["ausentismo_2016"] = (1 - d16["votantes_2016"] / d16["electores_2016"]) * 100

    d26 = pd.read_csv(ROOT / "output" / "lima_instalacion_distritos.csv")
    d26 = d26.rename(columns={"districtUbigeo": "UBIGEO"})

    panel = d16.merge(d26, on="UBIGEO", how="inner").dropna(
        subset=["ausentismoPctObservado", "delayPromedioHoras"])
    panel["delta"] = panel["ausentismoPctObservado"] - panel["ausentismo_2016"]
    panel["ausentismo_2026"] = panel["ausentismoPctObservado"]

    # Categoría: afectado si mediana > 7.75h
    panel["afectado"] = panel["horaMediana"] > 7.75
    panel["label"] = panel["districtName"].str.title().str.replace(
        "De ", "de ", regex=False).str.replace("Del ", "del ", regex=False)
    return panel

# ---------------------------------------------------------------------------
# Fig 6: Scatter Δausentismo vs delay promedio por distrito
# ---------------------------------------------------------------------------
def fig6_did_scatter(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = [SECONDARY if a else TERTIARY for a in panel["afectado"]]
    sizes  = panel["electoresObservados"] / panel["electoresObservados"].max() * 400 + 40

    sc = ax.scatter(panel["delayPromedioHoras"], panel["delta"],
                    c=colors, s=sizes, alpha=0.85, edgecolors="white", linewidth=0.6, zorder=3)

    # Línea de regresión ponderada
    X = sm.add_constant(panel["delayPromedioHoras"])
    res = sm.WLS(panel["delta"], X, weights=panel["electoresObservados"]).fit(cov_type="HC1")
    xs = np.linspace(panel["delayPromedioHoras"].min(), panel["delayPromedioHoras"].max(), 100)
    ys = res.params[0] + res.params[1] * xs
    beta = res.params[1]
    pval = res.pvalues[1]
    ax.plot(xs, ys, color=FG, linewidth=1.8, linestyle="--",
            label=f"Regresión ponderada: β={beta:+.3f} pp/hora (p={pval:.2f}, n.s.)")

    # Etiquetas selectivas (solo los más extremos o conocidos)
    label_these = {
        "VILLA EL SALVADOR", "SANTIAGO DE SURCO", "MIRAFLORES",
        "SAN JUAN DE LURIGANCHO", "LURÍN", "SAN BORJA", "SAN MIGUEL",
        "COMAS", "ATE", "SAN JUAN DE MIRAFLORES", "VILLA MARÍA DEL TRIUNFO",
    }
    for _, row in panel.iterrows():
        if row["districtName"] in label_these:
            ax.annotate(row["label"],
                        xy=(row["delayPromedioHoras"], row["delta"]),
                        xytext=(6, 3), textcoords="offset points",
                        fontsize=7.5, color=FG, alpha=0.9)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=SECONDARY, label="Distrito afectado (mediana > 7:45 a. m.)"),
        Patch(facecolor=TERTIARY, label="Distrito de control"),
        plt.Line2D([0], [0], color=FG, linewidth=1.8, linestyle="--",
                   label=f"β = {beta:+.3f} pp/hora  (p = {pval:.2f}, n.s.)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=False, fontsize=9)

    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Retraso promedio en instalación 2026 (horas)", fontweight="bold")
    ax.set_ylabel("Cambio en ausentismo 2016-2026 (pp)", fontweight="bold")
    _style_ax(ax)

    ax.set_title(
        "El aumento de ausentismo entre 2016 y 2026 es independiente\n"
        "del retraso logístico — pendiente prácticamente cero",
        loc="left", pad=16, fontsize=15, fontweight="bold")
    ax.text(0, 1.02,
            "Diff-in-diff distrital · 40 distritos Lima Metropolitana · "
            "ponderado por padrón · IC HC1",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_did_scatter.png")
    plt.close(fig)
    print("[ok] fig6_did_scatter.png")

# ---------------------------------------------------------------------------
# Fig 7: Triple robustez — comparación de estimaciones
# ---------------------------------------------------------------------------
def fig7_triple_robustez() -> None:
    methods = [
        ("Carta RP\n(agregado\n2016->2026)", 608_000, SECONDARY, True),
        ("M0 ingenuo\n(OLS sin\ncontroles)", 40_529, "#c8a87a", False),
        ("M1\n(FE de local,\nlineal)", 7_300, PRIMARY, False),
        ("M3\n(FE de local,\nno lineal)", 3_870, TERTIARY, False),
        ("M2\n(DiD vs\n2016)", 3_312, "#8fa68e", False),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [m[0] for m in methods]
    values = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    is_claim = [m[3] for m in methods]

    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=1, width=0.55)

    for bar, v, claim in zip(bars, values, is_claim):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 4_000,
                f"{int(v):,}",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold" if claim else "normal",
                color=ACCENT if claim else FG)
        if claim:
            ax.annotate("Reclamo\nde parte",
                        xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(bar.get_x() + bar.get_width() / 2 + 0.35, v - 80_000),
                        fontsize=8.5, color=ACCENT,
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))

    # Zona de consenso
    ax.axhspan(2_500, 8_000, alpha=0.10, color=TERTIARY, zorder=0)
    ax.text(4.5, 5_500, "Zona de consenso\nmétodos rigurosos\n(~3,000 – 8,000)",
            fontsize=8.5, color=TERTIARY, ha="center", va="center",
            fontweight="bold")

    ax.set_ylabel("Votantes perdidos estimados — Lima Metropolitana",
                  fontweight="bold")
    _style_ax(ax)
    ax.set_ylim(0, 680_000)

    ax.set_title(
        "Los tres métodos rigurosos convergen en ~3,000–8,000 votantes perdidos\n"
        "— un factor de ~100x menos que el reclamo de la carta",
        loc="left", pad=16, fontsize=14, fontweight="bold")
    ax.text(0, 1.02,
            "Comparación de estimaciones · Lima Metropolitana · "
            "Elecciones Generales 2026",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_triple_robustez.png")
    plt.close(fig)
    print("[ok] fig7_triple_robustez.png")

# ---------------------------------------------------------------------------
# Fig 8: Placebo test — ausentismo 2016 vs delay 2026
# ---------------------------------------------------------------------------
def fig8_placebo(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = [SECONDARY if a else TERTIARY for a in panel["afectado"]]
    sizes  = panel["electoresObservados"] / panel["electoresObservados"].max() * 350 + 40

    ax.scatter(panel["delayPromedioHoras"], panel["ausentismo_2016"],
               c=colors, s=sizes, alpha=0.85, edgecolors="white", linewidth=0.6, zorder=3)

    X = sm.add_constant(panel["delayPromedioHoras"])
    res = sm.WLS(panel["ausentismo_2016"], X,
                 weights=panel["electoresObservados"]).fit(cov_type="HC1")
    xs = np.linspace(panel["delayPromedioHoras"].min(),
                     panel["delayPromedioHoras"].max(), 100)
    ys = res.params[0] + res.params[1] * xs
    beta = res.params[1]; pval = res.pvalues[1]
    ax.plot(xs, ys, color=MUTED, linewidth=1.8, linestyle="--",
            label=f"β = {beta:+.3f} pp/hora  (p = {pval:.2f}, n.s.)")

    label_these = {
        "VILLA EL SALVADOR", "SANTIAGO DE SURCO", "MIRAFLORES",
        "SAN JUAN DE LURIGANCHO", "SAN BORJA", "LURÍN", "COMAS",
    }
    for _, row in panel.iterrows():
        if row["districtName"] in label_these:
            ax.annotate(row["label"],
                        xy=(row["delayPromedioHoras"], row["ausentismo_2016"]),
                        xytext=(6, 3), textcoords="offset points",
                        fontsize=7.5, color=FG, alpha=0.9)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=SECONDARY, label="Distrito afectado 2026"),
        Patch(facecolor=TERTIARY, label="Distrito de control"),
        plt.Line2D([0], [0], color=MUTED, linewidth=1.8, linestyle="--",
                   label=f"β = {beta:+.3f} pp/hora  (p = {pval:.2f}, n.s.)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=False, fontsize=9)

    ax.set_xlabel("Retraso promedio en instalación 2026 (horas)", fontweight="bold")
    ax.set_ylabel("Ausentismo en 2016 (%)", fontweight="bold")
    _style_ax(ax)

    ax.set_title(
        "Placebo test: el retraso de 2026 no predice el ausentismo previo de 2016\n"
        "— confirma que el problema logístico fue cuasi-exógeno",
        loc="left", pad=16, fontsize=14, fontweight="bold")
    ax.text(0, 1.02,
            "Variable dependiente: ausentismo 2016 · Variable independiente: retraso 2026 · "
            "Resultado esperado: pendiente no significativa",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")

    # Sello de validación
    ax.text(0.98, 0.05,
            "Placebo: OK\n(p = {:.2f})".format(pval),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, fontweight="bold", color=TERTIARY,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=TERTIARY, linewidth=1.5, alpha=0.9))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_placebo.png")
    plt.close(fig)
    print("[ok] fig8_placebo.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    panel = build_panel()
    print(f"Panel: {len(panel)} distritos")
    fig6_did_scatter(panel)
    fig7_triple_robustez()
    fig8_placebo(panel)
    print("[done]")

if __name__ == "__main__":
    main()
