"""
build_report.py

Pipeline completo que:
  1. Carga el dataset mesa-a-mesa y los resultados de M1/M3.
  2. Recalcula la imputación contrafactual usando los coeficientes no lineales.
  3. Genera gráficos del paper (matplotlib) con estética tipo referencia.
  4. Genera slides cuadradas para Twitter.

Outputs:
  output/figures/*.png
  output/slides/*.png
  output/contrafactual_revisado.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Paleta y estilo
# ---------------------------------------------------------------------------

BG = "#f5f1e8"
FG = "#2c2c2c"
MUTED = "#6c6c6c"
GRID = "#d8d2c4"

PRIMARY = "#6b8ca3"     # azul-gris
SECONDARY = "#c78585"   # rojo polvo
TERTIARY = "#8fa68e"    # verde salvia
QUATERNARY = "#d4a373"  # tan
ACCENT = "#a63a3a"      # rojo oscuro (líneas de referencia)
DARK = "#1e1e1e"

DISTRICT_PALETTE = [PRIMARY, SECONDARY, TERTIARY, QUATERNARY, "#8b7aa8", "#b89968"]

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": FG,
    "axes.titlecolor": FG,
    "text.color": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "font.family": "serif",
    "font.serif": ["Palatino", "Georgia", "DejaVu Serif", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CSV_MESAS = ROOT / "output" / "lima_instalacion_mesas.csv"
CSV_DISTRITOS = ROOT / "output" / "lima_instalacion_distritos.csv"
JSON_M1 = ROOT / "output" / "m1_resultados.json"
JSON_SUMMARY = ROOT / "output" / "lima_instalacion_summary.json"

FIG_DIR = ROOT / "output" / "figures"
SLIDE_DIR = ROOT / "output" / "slides"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SLIDE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_all():
    df = pd.read_csv(CSV_MESAS)
    df = df[df["codigoEstadoActa"] == "C"]
    df = df[df["installHour"].notna() & df["delayHours"].notna()]
    df = df[df["totalElectoresHabiles"] > 0]
    df = df[(df["installHour"] >= 4) & (df["installHour"] <= 16)]
    df = df[(df["ausentismoPct"] >= 0) & (df["ausentismoPct"] <= 100)]
    df = df.reset_index(drop=True)

    distritos = pd.read_csv(CSV_DISTRITOS)

    with JSON_M1.open() as f:
        m1 = json.load(f)
    with JSON_SUMMARY.open() as f:
        summary = json.load(f)

    return df, distritos, m1, summary


# ---------------------------------------------------------------------------
# Recalculo del contrafactual con M3 (bins no lineales)
# ---------------------------------------------------------------------------

def assign_bin(delay: float) -> str:
    if delay <= 0.5:
        return "bin_ref_0_05"
    if delay <= 1.0:
        return "bin_05_1"
    if delay <= 2.0:
        return "bin_1_2"
    if delay <= 3.0:
        return "bin_2_3"
    return "bin_3plus"


def recompute_contrafactual(df: pd.DataFrame, m1: dict) -> dict:
    """Imputación contrafactual usando los coeficientes de M3 (bins + FE)."""
    coefs = m1["modelo_C_nonlinear_fe"]["coeficientes_por_bin"]

    # Mapeo bin -> coeficiente (pp extras respecto al bin de referencia)
    bin_coef = {
        "bin_ref_0_05": 0.0,
        "bin_05_1": coefs["bin_05_1"]["coef_pp"],
        "bin_1_2": coefs["bin_1_2"]["coef_pp"],
        "bin_2_3": coefs["bin_2_3"]["coef_pp"],
        "bin_3plus": coefs["bin_3plus"]["coef_pp"],
    }

    # Versión conservadora: solo bins significativos
    bin_coef_conservative = {
        "bin_ref_0_05": 0.0,
        "bin_05_1": 0.0,   # no sig
        "bin_1_2": 0.0,    # no sig
        "bin_2_3": coefs["bin_2_3"]["coef_pp"],
        "bin_3plus": coefs["bin_3plus"]["coef_pp"],
    }

    df = df.copy()
    df["bin"] = df["delayHours"].apply(assign_bin)
    df["extra_pp_point"] = df["bin"].map(bin_coef)
    df["extra_pp_cons"] = df["bin"].map(bin_coef_conservative)
    df["extra_ausentes_point"] = (df["extra_pp_point"] / 100.0) * df["totalElectoresHabiles"]
    df["extra_ausentes_cons"] = (df["extra_pp_cons"] / 100.0) * df["totalElectoresHabiles"]

    total_point = df["extra_ausentes_point"].sum()
    total_cons = df["extra_ausentes_cons"].sum()

    # Prorrateo por candidato usando shares del CSV distrital
    distritos_df = pd.read_csv(CSV_DISTRITOS)
    shares = distritos_df.set_index("districtUbigeo")[[
        "share::RENOVACIÓN POPULAR",
        "share::PARTIDO DEL BUEN GOBIERNO",
        "share::JUNTOS POR EL PERÚ",
    ]]
    shares.columns = ["RP", "PBG", "JPP"]

    extra_by_distrito = df.groupby("districtUbigeo").agg(
        extra_ausentes_point=("extra_ausentes_point", "sum"),
        extra_ausentes_cons=("extra_ausentes_cons", "sum"),
        districtName=("districtName", "first"),
    )

    extra_by_distrito = extra_by_distrito.join(shares, how="left").fillna(0)

    for cand in ["RP", "PBG", "JPP"]:
        extra_by_distrito[f"lost_{cand}_point"] = (
            extra_by_distrito["extra_ausentes_point"] * extra_by_distrito[cand]
        )
        extra_by_distrito[f"lost_{cand}_cons"] = (
            extra_by_distrito["extra_ausentes_cons"] * extra_by_distrito[cand]
        )

    totals_point = {
        "RP": float(extra_by_distrito["lost_RP_point"].sum()),
        "PBG": float(extra_by_distrito["lost_PBG_point"].sum()),
        "JPP": float(extra_by_distrito["lost_JPP_point"].sum()),
    }
    totals_cons = {
        "RP": float(extra_by_distrito["lost_RP_cons"].sum()),
        "PBG": float(extra_by_distrito["lost_PBG_cons"].sum()),
        "JPP": float(extra_by_distrito["lost_JPP_cons"].sum()),
    }

    result = {
        "total_ausentes_extra_point": float(total_point),
        "total_ausentes_extra_conservador": float(total_cons),
        "candidatos_point": totals_point,
        "candidatos_conservador": totals_cons,
        "mesas_por_bin": df["bin"].value_counts().to_dict(),
        "electores_por_bin": df.groupby("bin")["totalElectoresHabiles"].sum().to_dict(),
    }

    out_path = ROOT / "output" / "contrafactual_revisado.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[ok] contrafactual revisado guardado en {out_path}")
    return result


# ---------------------------------------------------------------------------
# Helpers de gráficos
# ---------------------------------------------------------------------------

def _format_hour(h: float) -> str:
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh += 1
        mm = 0
    suffix = "a. m." if hh < 12 else "p. m."
    hh12 = hh if hh <= 12 else hh - 12
    if hh12 == 0:
        hh12 = 12
    return f"{hh12}:{mm:02d} {suffix}"


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(True, axis="y", alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figura 1: Histograma de horas de instalación por distrito afectado
# ---------------------------------------------------------------------------

def fig_histograma(df: pd.DataFrame, summary: dict) -> None:
    affected = [
        ("VILLA EL SALVADOR", 140141),
        ("SAN JUAN DE MIRAFLORES", 140136),
        ("SANTIAGO DE SURCO", 140130),
        ("SAN MIGUEL", 140127),
        ("MIRAFLORES", 140115),
        ("SAN BORJA", 140140),
    ]
    affected_ubis = [u for _, u in affected]
    sub_all = df[df["districtUbigeo"].isin(affected_ubis)]
    pct_late = (sub_all["installHour"] > 7.0).mean() * 100
    n_affected = len(sub_all)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bins = np.arange(4.0, 16.25, 0.25)
    bottoms = np.zeros(len(bins) - 1)

    medians = {}
    for i, (name, ubi) in enumerate(affected):
        sub = df[df["districtUbigeo"] == ubi]
        if sub.empty:
            continue
        hist, _ = np.histogram(sub["installHour"], bins=bins)
        color = DISTRICT_PALETTE[i % len(DISTRICT_PALETTE)]
        ax.bar(
            bins[:-1], hist, width=0.23,
            bottom=bottoms, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.4,
            label=f"{name.title()} (mediana {_format_hour(sub['installHour'].median())})",
        )
        bottoms += hist
        medians[name] = sub["installHour"].median()

    ax.axvline(7.0, color=ACCENT, linestyle="--", linewidth=1.6,
               label="7:00 a. m. — hora legal de apertura", zorder=0)

    ax.set_xlim(4, 14)
    ax.set_xticks(range(4, 15))
    ax.set_xticklabels([_format_hour(h).replace(" ", "\n") for h in range(4, 15)],
                       fontsize=8)
    ax.set_xlabel("Hora de instalación de mesa", fontweight="bold")
    ax.set_ylabel("Número de mesas")
    _style_ax(ax)

    ax.set_title(
        f"El {pct_late:.0f}% de las mesas en 6 distritos afectados se instaló\n"
        "después de las 7:00 a. m., hora legal de inicio del sufragio",
        loc="left", pad=16, fontsize=15, fontweight="bold",
    )
    ax.text(0, 1.02,
            f"Distribución de horas de instalación · Elecciones Generales 2026 · {n_affected:,} mesas de 6 distritos afectados",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")

    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_histograma.png")
    plt.close(fig)
    print("[ok] fig1_histograma.png")


# ---------------------------------------------------------------------------
# Figura 2: Scatter con DOS líneas de regresión (ingenua vs within)
# ---------------------------------------------------------------------------

def fig_naive_vs_fe(df: pd.DataFrame, m1: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    sample = df.sample(min(6000, len(df)), random_state=7)
    ax.scatter(
        sample["installHour"], sample["ausentismoPct"],
        s=7, alpha=0.18, color=PRIMARY, edgecolors="none",
    )

    # Línea ingenua (Modelo A)
    a = m1["modelo_A_naive"]
    xs = np.linspace(5, 14, 50)
    naive_line = a["intercept"] + a["slope_pp_per_hour"] * (xs - 7.0)
    ax.plot(xs, naive_line, color=ACCENT, linewidth=2.8, linestyle="-",
            label=f"OLS ingenuo: +{a['slope_pp_per_hour']:.2f} pp/hora")

    # Línea within (Modelo B) — anclada en el ausentismo promedio a las 7am
    b = m1["modelo_B_fixed_effects"]
    anchor_y = df.loc[(df["installHour"] >= 6.8) & (df["installHour"] <= 7.2),
                       "ausentismoPct"].mean()
    within_line = anchor_y + b["slope_pp_per_hour"] * (xs - 7.0)
    ax.plot(xs, within_line, color=TERTIARY, linewidth=2.8, linestyle="-",
            label=f"Efectos fijos de local: +{b['slope_pp_per_hour']:.2f} pp/hora")

    ax.axvline(7.0, color=MUTED, linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlim(5, 14)
    ax.set_ylim(0, 55)
    ax.set_xticks(range(5, 15))
    ax.set_xticklabels([_format_hour(h).replace(" ", "\n") for h in range(5, 15)],
                       fontsize=8)
    ax.set_xlabel("Hora de instalación de mesa", fontweight="bold")
    ax.set_ylabel("Ausentismo (%)", fontweight="bold")
    _style_ax(ax)

    ax.set_title(
        "Al controlar por local de votación, el efecto del retraso\n"
        "se reduce casi 90% — el resto era confounding",
        loc="left", pad=16, fontsize=15, fontweight="bold",
    )
    ax.text(0, 1.02,
            "Comparación de especificaciones · Lima Metropolitana · 19,812 mesas · 1,684 locales con ≥2 mesas",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")

    ax.legend(loc="upper left", frameon=False, fontsize=10)

    # Anotación
    ax.annotate(
        f"Pasar de β=0.88 (ingenuo)\na β=0.09 (con FE)\nno es casualidad:\nconfounders de distrito/NSE\nexplicaban la mayoría.",
        xy=(12.5, 28), fontsize=9, color=FG,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=MUTED, linewidth=0.5, alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_naive_vs_fe.png")
    plt.close(fig)
    print("[ok] fig2_naive_vs_fe.png")


# ---------------------------------------------------------------------------
# Figura 3: Efecto no lineal por bins con intervalos de confianza
# ---------------------------------------------------------------------------

def fig_bins(m1: dict) -> None:
    coefs = m1["modelo_C_nonlinear_fe"]["coeficientes_por_bin"]
    bins_order = ["bin_05_1", "bin_1_2", "bin_2_3", "bin_3plus"]
    labels = ["0.5 – 1 h", "1 – 2 h", "2 – 3 h", "3+ h"]

    values = [coefs[b]["coef_pp"] for b in bins_order]
    lows = [coefs[b]["ci95_low"] for b in bins_order]
    highs = [coefs[b]["ci95_high"] for b in bins_order]
    errs_lo = [v - l for v, l in zip(values, lows)]
    errs_hi = [h - v for v, h in zip(values, highs)]
    ns = [coefs[b]["n_mesas_en_bin"] for b in bins_order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [
        "#cccccc" if (l < 0 < h) else PRIMARY
        for l, h in zip(lows, highs)
    ]
    colors = [PRIMARY if v == PRIMARY else "#b8bfa8" for v in colors]
    # Re-mark: no significativos en gris, significativos en azul
    colors = []
    for l, h in zip(lows, highs):
        if l > 0 or h < 0:
            colors.append(PRIMARY)
        else:
            colors.append("#bbb5a3")

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1)
    ax.errorbar(labels, values, yerr=[errs_lo, errs_hi],
                fmt="none", ecolor=DARK, capsize=5, capthick=1.2, linewidth=1.2)

    ax.axhline(0, color=MUTED, linewidth=1)

    for i, (bar, v, n) in enumerate(zip(bars, values, ns)):
        sig = (lows[i] > 0) or (highs[i] < 0)
        mark = "significativo" if sig else "n. s."
        ax.text(bar.get_x() + bar.get_width() / 2,
                highs[i] + 0.04,
                f"+{v:.2f} pp\n{mark}\nN={n:,}",
                ha="center", va="bottom", fontsize=9,
                color=FG if sig else MUTED,
                fontweight="bold" if sig else "normal")

    ax.set_ylabel("Exceso de ausentismo (pp) vs. mesas abiertas a tiempo",
                  fontweight="bold")
    ax.set_xlabel("Retraso en la instalación respecto a las 7:00 a. m.",
                  fontweight="bold")
    ax.set_ylim(-0.3, max(highs) + 0.45)
    _style_ax(ax)

    ax.set_title(
        "Solo los retrasos mayores a 2 horas tienen efecto\n"
        "estadísticamente detectable sobre el ausentismo",
        loc="left", pad=16, fontsize=15, fontweight="bold",
    )
    ax.text(0, 1.02,
            "Coeficientes de bins · Modelo con efectos fijos de local · IC 95% clusterizados por local",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_bins.png")
    plt.close(fig)
    print("[ok] fig3_bins.png")


# ---------------------------------------------------------------------------
# Figura 4: Comparación contrafactual ingenuo vs revisado
# ---------------------------------------------------------------------------

def fig_contrafactual(summary: dict, contra: dict) -> None:
    naive = summary["candidateLossesEstimated"]
    revised = contra["candidatos_point"]
    conservative = contra["candidatos_conservador"]

    cands = ["RP", "PBG", "JPP"]
    cand_names = ["Renovación Popular", "Partido del Buen Gobierno", "Juntos por el Perú"]
    naive_vals = [
        naive["RENOVACIÓN POPULAR"],
        naive["PARTIDO DEL BUEN GOBIERNO"],
        naive["JUNTOS POR EL PERÚ"],
    ]
    revised_vals = [revised[c] for c in cands]
    cons_vals = [conservative[c] for c in cands]

    x = np.arange(len(cands))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 6))

    b1 = ax.bar(x - width, naive_vals, width, color=SECONDARY,
                edgecolor="white", label="Ingenuo (β=0.73)")
    b2 = ax.bar(x, revised_vals, width, color=PRIMARY,
                edgecolor="white", label="M3 no lineal")
    b3 = ax.bar(x + width, cons_vals, width, color=TERTIARY,
                edgecolor="white", label="M3 conservador (solo sig.)")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 50,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(cand_names, fontsize=10)
    ax.set_ylabel("Votos perdidos estimados", fontweight="bold")
    _style_ax(ax)

    ax.set_title(
        "El análisis ingenuo infla los votos perdidos ~10x\n"
        "respecto a una estimación con efectos fijos",
        loc="left", pad=16, fontsize=15, fontweight="bold",
    )
    ax.text(0, 1.02,
            "Imputación contrafactual por candidato · Lima Metropolitana · Supuesto de preferencias homogéneas",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")

    ax.legend(loc="upper right", frameon=False, fontsize=9)

    total_naive = sum(naive_vals)
    total_rev = sum(revised_vals)
    total_cons = sum(cons_vals)
    ax.text(0.02, 0.78,
            f"Total ausentes extra\n"
            f"  • Ingenuo: {int(summary['totalExtraAusentesEstimated']):,}\n"
            f"  • M3 punto: {int(contra['total_ausentes_extra_point']):,}\n"
            f"  • M3 conservador: {int(contra['total_ausentes_extra_conservador']):,}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                      edgecolor=MUTED, linewidth=0.5, alpha=0.9),
            verticalalignment="top")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_contrafactual.png")
    plt.close(fig)
    print("[ok] fig4_contrafactual.png")


# ---------------------------------------------------------------------------
# Figura 5: Ranking de distritos afectados vs control
# ---------------------------------------------------------------------------

def fig_ranking(distritos: pd.DataFrame) -> None:
    d = distritos.dropna(subset=["horaMediana"]).copy()
    d = d.sort_values("horaMediana", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 7.5))

    # Códigos numéricos > 7.75 = afectados
    colors = [SECONDARY if h > 7.75 else TERTIARY for h in d["horaMediana"]]

    ax.barh(d["districtName"].str.title(), d["horaMediana"],
            color=colors, edgecolor="white", linewidth=0.5)

    ax.axvline(7.0, color=ACCENT, linestyle="--", linewidth=1.2,
               label="7:00 a. m. (hora legal)")
    ax.axvline(7.75, color=MUTED, linestyle=":", linewidth=1,
               label="7:45 a. m. (umbral de 'afectado')")

    for i, (name, h) in enumerate(zip(d["districtName"], d["horaMediana"])):
        ax.text(h + 0.05, i, _format_hour(h), va="center", fontsize=8, color=FG)

    ax.set_xlim(6.5, 10.5)
    ax.set_xticks(np.arange(6.5, 10.51, 0.5))
    ax.set_xticklabels([_format_hour(h) for h in np.arange(6.5, 10.51, 0.5)],
                       fontsize=8)
    ax.set_xlabel("Mediana de hora de instalación de mesas", fontweight="bold")
    _style_ax(ax)

    ax.set_title(
        "No fue 'Lima Sur' — el retraso afectó un corredor mixto\n"
        "de Lima Moderna y Lima Sur",
        loc="left", pad=16, fontsize=14, fontweight="bold",
    )
    ax.text(0, 1.02,
            "Top 20 distritos de Lima Metropolitana por mediana de hora de instalación",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")

    # Leyenda custom
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SECONDARY, label="Afectado (mediana > 7:45)"),
        Patch(facecolor=TERTIARY, label="Control (mediana ≤ 7:45)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_ranking.png")
    plt.close(fig)
    print("[ok] fig5_ranking.png")


# ---------------------------------------------------------------------------
# Slides para Twitter (1080x1080)
# ---------------------------------------------------------------------------

def _new_slide():
    fig, ax = plt.subplots(figsize=(10, 10), dpi=108)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _slide_header(ax, tag: str):
    ax.text(0.08, 0.93, tag, fontsize=11, color=MUTED, fontweight="bold")
    ax.plot([0.08, 0.22], [0.915, 0.915], color=ACCENT, linewidth=2.5)


def _slide_footer(ax, num: int, total: int):
    ax.text(0.08, 0.05,
            "Análisis: retraso en instalación de mesas · Elecciones 2026 Lima",
            fontsize=9, color=MUTED, style="italic")
    ax.text(0.92, 0.05, f"{num}/{total}", fontsize=10, color=MUTED, ha="right")


def slide_1_hook(contra, summary):
    fig, ax = _new_slide()
    _slide_header(ax, "HALLAZGO PRINCIPAL")

    ax.text(0.08, 0.78,
            "El problema logístico",
            fontsize=28, fontweight="bold", color=FG)
    ax.text(0.08, 0.71,
            "sí existió, pero no cambió",
            fontsize=28, fontweight="bold", color=FG)
    ax.text(0.08, 0.64,
            "el resultado de la elección",
            fontsize=28, fontweight="bold", color=ACCENT)
    ax.text(0.08, 0.57, "en Lima.",
            fontsize=28, fontweight="bold", color=ACCENT)

    naive_total = int(summary["totalExtraAusentesEstimated"])
    revised_total = int(contra["total_ausentes_extra_point"])

    ax.text(0.08, 0.44,
            "Análisis ingenuo (sin controles):",
            fontsize=13, color=MUTED)
    ax.text(0.08, 0.38,
            f"~{naive_total:,} votantes perdidos",
            fontsize=22, fontweight="bold", color=SECONDARY)

    ax.text(0.08, 0.28,
            "Análisis con efectos fijos de local:",
            fontsize=13, color=MUTED)
    ax.text(0.08, 0.22,
            f"~{revised_total:,} votantes perdidos",
            fontsize=22, fontweight="bold", color=TERTIARY)

    ax.text(0.08, 0.13,
            "Una diferencia de ~10x. El 90% del 'efecto' ingenuo\nera confounding entre distritos.",
            fontsize=11, color=FG, style="italic")

    _slide_footer(ax, 1, 5)
    fig.savefig(SLIDE_DIR / "slide1_hook.png", facecolor=BG)
    plt.close(fig)
    print("[ok] slide1_hook.png")


def slide_2_geografia():
    import matplotlib.image as mpimg
    img = mpimg.imread(FIG_DIR / "fig1_histograma.png")

    fig, ax = _new_slide()
    _slide_header(ax, "02 · GEOGRAFÍA DEL PROBLEMA")

    ax.text(0.08, 0.85,
            "No fue Lima Sur solamente.",
            fontsize=22, fontweight="bold", color=FG)
    ax.text(0.08, 0.80,
            "Surco, Miraflores, San Miguel y San Borja",
            fontsize=14, color=FG)
    ax.text(0.08, 0.76,
            "también tuvieron retrasos sistemáticos.",
            fontsize=14, color=FG)

    ax_img = fig.add_axes([0.06, 0.12, 0.88, 0.58])
    ax_img.imshow(img)
    ax_img.axis("off")

    _slide_footer(ax, 2, 5)
    fig.savefig(SLIDE_DIR / "slide2_geografia.png", facecolor=BG)
    plt.close(fig)
    print("[ok] slide2_geografia.png")


def slide_3_metodologia(m1):
    fig, ax = _new_slide()
    _slide_header(ax, "03 · EL GIRO METODOLÓGICO")

    a = m1["modelo_A_naive"]
    b = m1["modelo_B_fixed_effects"]

    ax.text(0.08, 0.85,
            "El 90% del efecto era",
            fontsize=26, fontweight="bold", color=FG)
    ax.text(0.08, 0.79,
            "confounding entre distritos.",
            fontsize=26, fontweight="bold", color=FG)

    # Barras comparativas
    ax.text(0.08, 0.63, "OLS ingenuo", fontsize=12, color=MUTED)
    ax.add_patch(Rectangle((0.08, 0.52), 0.70 * (a["slope_pp_per_hour"] / 1.0),
                           0.08, facecolor=SECONDARY, edgecolor="none"))
    ax.text(0.08 + 0.70 * a["slope_pp_per_hour"] + 0.02, 0.555,
            f"+{a['slope_pp_per_hour']:.2f} pp/h",
            fontsize=14, fontweight="bold", color=SECONDARY, va="center")

    ax.text(0.08, 0.43, "Con efectos fijos de local", fontsize=12, color=MUTED)
    ax.add_patch(Rectangle((0.08, 0.32), 0.70 * (b["slope_pp_per_hour"] / 1.0),
                           0.08, facecolor=TERTIARY, edgecolor="none"))
    ax.text(0.08 + 0.70 * b["slope_pp_per_hour"] + 0.02, 0.355,
            f"+{b['slope_pp_per_hour']:.2f} pp/h",
            fontsize=14, fontweight="bold", color=TERTIARY, va="center")

    ax.text(0.08, 0.22,
            "Comparando mesas DENTRO del mismo local\n"
            "(mismo barrio, NSE, clima, electorado), el efecto\n"
            "verdadero del retraso cae de 0.88 a 0.09 pp/hora.",
            fontsize=11, color=FG)

    ax.text(0.08, 0.12,
            "Esto es lo que pasa cuando separas\ncorrelación de causalidad.",
            fontsize=10, color=MUTED, style="italic")

    _slide_footer(ax, 3, 5)
    fig.savefig(SLIDE_DIR / "slide3_metodologia.png", facecolor=BG)
    plt.close(fig)
    print("[ok] slide3_metodologia.png")


def slide_4_no_lineal():
    import matplotlib.image as mpimg
    img = mpimg.imread(FIG_DIR / "fig3_bins.png")

    fig, ax = _new_slide()
    _slide_header(ax, "04 · EFECTO NO LINEAL")

    ax.text(0.08, 0.85,
            "Un retraso de 1 hora no tuvo",
            fontsize=22, fontweight="bold", color=FG)
    ax.text(0.08, 0.80,
            "efecto detectable. Solo los retrasos",
            fontsize=22, fontweight="bold", color=FG)
    ax.text(0.08, 0.75,
            "≥ 2 horas movieron la aguja.",
            fontsize=22, fontweight="bold", color=ACCENT)

    ax_img = fig.add_axes([0.06, 0.12, 0.88, 0.55])
    ax_img.imshow(img)
    ax_img.axis("off")

    _slide_footer(ax, 4, 5)
    fig.savefig(SLIDE_DIR / "slide4_no_lineal.png", facecolor=BG)
    plt.close(fig)
    print("[ok] slide4_no_lineal.png")


def slide_5_conclusion(contra):
    fig, ax = _new_slide()
    _slide_header(ax, "05 · CONCLUSIÓN")

    ax.text(0.08, 0.85,
            "La historia honesta",
            fontsize=26, fontweight="bold", color=FG)

    bullets = [
        ("+", "El retraso fue real y verificable: ~2,500 mesas en Lima\nabrieron con 2+ horas de retraso.", TERTIARY),
        ("+", "El efecto causal sobre la participación es\nestadísticamente significativo.", TERTIARY),
        ("−", "La magnitud NO alcanza para alterar el resultado\npresidencial en Lima.", SECONDARY),
        ("+", "Los ~3,000 votantes perdidos están concentrados\nen las mesas con retrasos extremos.", TERTIARY),
    ]

    y = 0.72
    for mark, text, color in bullets:
        ax.text(0.08, y, mark, fontsize=20, color=color, fontweight="bold")
        ax.text(0.14, y + 0.005, text, fontsize=12, color=FG)
        y -= 0.11

    ax.text(0.08, 0.22,
            "Vale la pena denunciar el problema logístico",
            fontsize=12, color=FG, fontweight="bold")
    ax.text(0.08, 0.175,
            "sin exagerar su impacto electoral.",
            fontsize=12, color=FG, fontweight="bold")

    ax.text(0.08, 0.11,
            "Metodología: efectos fijos de local de votación +\nbins no lineales · Lima · 19,812 mesas · 1,684 locales",
            fontsize=9, color=MUTED, style="italic")

    _slide_footer(ax, 5, 5)
    fig.savefig(SLIDE_DIR / "slide5_conclusion.png", facecolor=BG)
    plt.close(fig)
    print("[ok] slide5_conclusion.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df, distritos, m1, summary = load_all()
    print(f"[load] {len(df):,} mesas | {df['codigoLocalVotacion'].nunique():,} locales")

    contra = recompute_contrafactual(df, m1)

    print("\n-- Figuras del paper --")
    fig_histograma(df, summary)
    fig_naive_vs_fe(df, m1)
    fig_bins(m1)
    fig_contrafactual(summary, contra)
    fig_ranking(distritos)

    print("\n-- Slides de Twitter --")
    slide_1_hook(contra, summary)
    slide_2_geografia()
    slide_3_metodologia(m1)
    slide_4_no_lineal()
    slide_5_conclusion(contra)

    print("\n[done]")


if __name__ == "__main__":
    main()
