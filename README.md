# Dashboard Electoral Perú 2026 — Primera Vuelta

Seguimiento en tiempo real de los resultados de la primera vuelta presidencial peruana (14 de abril de 2026), con proyecciones bayesianas y análisis causal del impacto de los retrasos en la instalación de mesas de votación.

**→ [Ver dashboard en vivo](https://onpe-2026-dashboard.vercel.app)**

---

## Qué hace este proyecto

### 1. Dashboard en tiempo real

El dashboard consume la API pública de ONPE cada 15 minutos (vía un servidor VPS), genera proyecciones bayesianas del resultado final, y publica automáticamente a Vercel.

Muestra:
- Porcentaje de actas contadas y votos válidos al momento
- Resultados actuales de los 4 candidatos principales
- **Proyección bayesiana** del resultado al 100% de actas, con intervalos de confianza al 95%
- Probabilidad de cada candidato de pasar a segunda vuelta
- Evolución histórica del conteo (gráfico de trayectoria)
- Desglose por región, Lima, y voto extranjero

### 2. Análisis causal: retrasos en instalación de mesas

El 14 de abril de 2026, fallas logísticas en Lima (especialmente Lima Sur) causaron que numerosas mesas abrieran significativamente después de las 7:00 a.m. Este análisis estima el impacto causal de esos retrasos sobre la participación electoral.

**Modelos estimados:**

| Modelo | Especificación | Resultado |
|--------|---------------|-----------|
| M1 — Efectos fijos por local de votación | Variación *within* local (mesas del mismo local con distintos tiempos de apertura) | **0.09 pp menos participación por hora de retraso** |
| M2 — Diferencias en diferencias (2016 vs 2026) | Cambio en abstención distrital vs nivel de retraso | 0.06 pp/hora (no significativo) |
| M3 — Efectos fijos no lineales | Bins por magnitud de retraso | El daño se acelera a partir de 2 horas (0.3–0.5 pp) |

**Contrafactual:** Se estiman ~3,870 votos perdidos en Lima atribuibles a los retrasos de instalación, distribuidos entre los candidatos según la composición del voto de cada mesa afectada.

---

## Estructura del repositorio

```
├── dashboard.html              # Dashboard principal (sitio estático)
├── dashboard_data.json         # Snapshot de datos para el dashboard
├── onpe_dashboard.py           # Generador del dashboard HTML
│
├── onpe_proyeccion.py          # Modelo base (nivel región)
├── onpe_proyeccion_v2.py       # Modelo por provincia (273 provincias)
├── onpe_proyeccion_v3.py       # Modelo por distrito (2102 distritos)
│
├── lima_instalacion_analysis.py # Extracción de tiempos de PDFs de ONPE
├── m1_efectos_fijos.py          # Modelo M1: efectos fijos por local
├── m2_diff_in_diff.py           # Modelo M2: diff-in-diff 2016–2026
├── calibrar_2021.py             # Calibración con EG2021 como verdad de tierra
│
├── output/
│   ├── m1_resultados.json       # Coeficientes y CI del modelo M1
│   ├── m2_resultados.json       # Resultados M2 + tests de placebo
│   ├── contrafactual_revisado.json # Votos perdidos por candidato
│   └── figures/                 # Figuras del paper (8 PNG)
│
├── paper/index.html             # Análisis complementario publicado
├── worker/                      # Cloudflare Worker (proxy anti-bloqueo de IP)
└── scripts/vps_refresh.sh       # Script de actualización automática en VPS
```

---

## Modelo de proyección

Se usa un modelo **Dirichlet-Multinomial bayesiano** por unidad geográfica:

- **Prior:** padrón electoral (RENIEC) por circunscripción
- **Likelihood:** votos observados hasta el momento
- **Posterior:** ~1200 muestras MCMC para estimar la distribución del resultado final
- **Corrección de sesgo temporal:** las circunscripciones que reportan primero tienen composición distinta a las que reportan tarde; el modelo ajusta por esto

Las versiones v2 y v3 aumentan la granularidad a provincia y distrito respectivamente, mejorando la precisión en zonas con composición electoral heterogénea.

---

## Arquitectura de actualización automática

```
Cron VPS (cada 15 min)
  → scripts/vps_refresh.sh
  → python onpe_dashboard.py
     → API ONPE (directo)
     → si IP bloqueada → Cloudflare Worker proxy
  → git push → Vercel redespliega automáticamente
```

El dashboard en Vercel tiene cache desactivado (`no-store`) para garantizar que los visitantes siempre vean la versión más reciente.

---

## Dependencias

```
numpy
requests
pandas
statsmodels
matplotlib
pypdf
```

```bash
pip install -r requirements.txt
```

---

## Nota sobre los datos

Los archivos de estado pesados (`onpe_state_district.json`, ~393 MB) y los caches de extracción de PDFs se excluyen del repositorio. Se regeneran localmente al correr los scripts de análisis.
