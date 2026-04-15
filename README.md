# ONPE 2026 Dashboard

Dashboard y scripts de análisis local para seguimiento de resultados de la primera vuelta 2026 en Perú.

## Qué incluye

- `dashboard.html`: dashboard principal listo para publicar como sitio estático.
- `paper/`: análisis complementario sobre instalación de mesas y participación en Lima.
- `onpe_dashboard.py`: generador del dashboard.
- `onpe_proyeccion.py`, `onpe_proyeccion_v2.py`, `onpe_proyeccion_v3.py`: variantes del modelo de proyección.

## Publicación

El proyecto está preparado para publicarse como sitio estático en Vercel usando `vercel.json`, con la raíz apuntando al dashboard.

## Nota

Los estados pesados y logs locales de extracción se excluyen del repositorio público para mantener el deploy limpio y evitar subir artefactos regenerables.
