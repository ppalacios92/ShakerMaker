# `DRM_QA_Data` horizontal components turn NaN partway through the record

**Síntoma**: en un `.h5drm` generado con `DRMHDF5StationListWriter` (modo `progressive`), el
grupo `/DRM_QA_Data` tiene `velocity`/`displacement`/`acceleration` válidos en las 3 componentes
solo hasta cierto paso de tiempo — de ahí en adelante, las dos componentes **horizontales**
(`e`, `n` — filas 0 y 1) quedan en `NaN` hasta el final del registro. La componente **vertical**
(`z`, fila 2) está limpia todo el registro.

## Reproducción

`examples/08_drm/drm_loh1.py` (SCEC LOH.1 benchmark, estación `Centro`), corrido tal cual está
en el repo. Output real (`examples/08_drm/drm_loh1_output/drm_Centro_sta0.h5drm`, generado
2026-08-29):

```python
import h5py, numpy as np
with h5py.File('drm_loh1_output/drm_Centro_sta0.h5drm', 'r') as f:
    vel = f['DRM_QA_Data/velocity'][...]   # shape (3, 2400)

np.isnan(vel).sum()          # 4674 / 7200
np.isnan(vel[0]).sum()       # 2337 -- componente 'e' (east)
np.isnan(vel[1]).sum()       # 2337 -- componente 'n' (north)
np.isnan(vel[2]).sum()       # 0    -- componente 'z' (vertical), limpia
np.where(np.isnan(vel[0]))[0][0]   # 63  -> t = 63*dt = 63*0.005 = 0.315 s
```

Mismo patrón, mismos índices exactos (63/2337/2337/0), en una copia usada río abajo en otro
proyecto (`.../CERNProject/.../drm_load_pattern/loh1_reference.h5drm`) — confirma que no es
corrupción al copiar el archivo, nace así en la corrida real de ShakerMaker.

`displacement` y `acceleration` tienen el mismo patrón (mismos índices) porque se derivan de
`velocity` en el mismo paso de escritura (ver abajo).

## Dónde escribe esto el código

`shakermaker/slw_extensions/drmhdf5stationlistwriter.py`, `_write_station_progressive_drm`:

```python
ve = _interpolate(t, ee, t_final)
vn = _interpolate(t, nn, t_final)
vz = _interpolate(t, zz, t_final)
...
grp['velocity'][row,     :] = ve   # row=0 -> e (east)
grp['velocity'][row + 1, :] = vn   # row=1 -> n (north)
grp['velocity'][row + 2, :] = vz   # row=2 -> z (vertical)
```
(`displacement`/`acceleration` se derivan de `ve`/`vn`/`vz` por integración/diferenciación
trapezoidal justo después — cualquier NaN en `ve`/`vn` se propaga automáticamente a las tres.)

`_interpolate` es un `scipy.interpolate.interp1d` con `bounds_error=False,
fill_value=(yold[0], yold[-1])` — clampa fuera de rango, **no genera NaN por sí solo** si
`yold` (acá, `ee`/`nn`, la respuesta cruda de `station.get_response()`) está limpio en todo su
rango muestreado. Que la interpolación produzca NaN implica que `ee`/`nn` YA traían NaN antes de
llegar a este writer, en algún punto de su propio muestreo — la interpolación solo lo está
propagando a los puntos vecinos de `t_final`.

## Hipótesis (no confirmada a fondo — falta trazar `station.get_response()`)

El patrón (horizontal roto, vertical limpio, mismo índice de corte en las dos componentes
horizontales) apunta a algo específico de cómo se sintetiza la respuesta horizontal vs. vertical
en la función de Green (probablemente en el motor FK, río arriba de este writer, no en el writer
mismo) — por ejemplo, una división por un término que se anula o cambia de signo cerca de
t=0.315s solo en la combinación radial/tangencial que da `e`/`n`, mientras que la vertical no
pasa por esa misma operación. No rastreado más allá de este punto — el siguiente paso natural es
inspeccionar `station.get_response()` y de ahí hacia atrás hasta la síntesis FK, con y sin el
mismo `dt`/`nfft`/`dk` de `drm_loh1.py` para ver si el corte se mueve con esos parámetros (lo que
apuntaría a un problema numérico dependiente de resolución, no a un bug de índice fijo).

## Alcance / impacto

Confirmado que **`/DRM_Data`** (la grilla real que consumen los DRM boxes para excitar un
modelo FE) está completamente limpia (0 NaN en 58.8M valores) en el mismo archivo — el problema
está aislado a `/DRM_QA_Data`, la estación de control de referencia/visualización. No afecta
resultados de FEM que ya se corrieron contra este `.h5drm`; sí rompe cualquier comparación visual
horizontal contra la estación QA más allá de t≈0.315 s.
