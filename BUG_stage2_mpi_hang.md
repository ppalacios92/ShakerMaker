# Stage 2 (`run_fast`) cuelga el job MPI aunque el cómputo ya terminó

**Síntoma**: en una campaña de convolución (`model.run_nearest(stage=2, ...)`), el
cómputo científico termina correctamente y rápido, y el `.h5` de salida queda completo
y válido (verificado con `h5py`, `writer.close()` ya ejecutado), pero el job SLURM
**no libera sus nodos**. El log se detiene siempre en el mismo punto exacto, sin nada
más después, minutos u horas más tarde:

```
ShakerMaker Run (Stage 2 - OP) done. Total time: 250.11 s
--------------------------------------------------

Performance statistics (all MPI processes):
```
*(fin del log — nunca llegan las 5 líneas de estadísticas que deberían seguir)*

Reproducido 2/2 veces en la misma campaña (`crust01/segment_01` y `crust01/segment_02`,
10 nodos × 16 tareas = 160 ranks MPI, 1 sola estación activa, 32768 subfallas). Ver
evidencia completa en
`CERNProject/10. ShakermakerModel/04_adjust_models/bug_shakermaker/`.

## Mecanismo

`_print_perf_stats` (`shakermaker/shakermaker.py:134-149`) llama `comm.Reduce(...)`
sin condición, en todos los ranks — una colectiva **bloqueante**. La cabecera
`"Performance statistics..."` la imprime solo `rank==0`, **antes** del primer
`Reduce`: el log confirma que rank 0 llegó ahí y quedó esperando. Si un solo rank de
los 160 no llega a la misma línea, todos los demás esperan para siempre.

Dentro de `run_fast`, antes del fix, el único `try/except` + `comm.Abort()` protegía
*solo* `station.add_to_response(...)`. Todo lo demás — apertura de los 2 `.h5` y
lectura de metadatos antes del loop (ejecutado igual por los 160 ranks contra los
mismos 2 archivos, con `HDF5_USE_FILE_LOCKING=FALSE`), la lectura de `tdata`/
`deepcopy`/`split_at_depth`/`_call_core_fast`/`convolve` dentro del loop del rank
"dueño" de cada estación, y los `Send`/`Recv` MPI — no tenía ninguna protección.
Cualquier excepción ahí mataba ese rank en silencio, sin `Abort()`, y colgaba a todos
los demás en el `Reduce`.

**Dato clave para descartar la hipótesis obvia**: con `nstations=1` (como en esta
campaña), `owner = i_station % nprocs` es *siempre* rank 0, y el log muestra
`rank=0 sta 1/1 (100.0%)` seguido de `"...done. Total time..."` — es decir, el rank
0/dueño sí completó todo el loop de subfallas sin excepción. La causa no está en ese
loop del dueño para este caso concreto, sino, lo más probable, en el bloque
compartido de apertura/lectura HDF5 que ejecutan por igual los 159 ranks ociosos,
bajo contención de E/S de un filesystem compartido con ~160 aperturas concurrentes de
los mismos 2 archivos — un fallo transitorio ahí, en cualquiera de esos ranks, nunca
se hacía visible ni abortaba el job.

## Fix

Se extendió el mismo patrón `try/except Exception: traceback.print_exc(); comm.Abort()`
que ya usaba el código, a los 4 puntos sin protección de `run_fast`: el bloque de
apertura/lectura previo al loop, la lectura de `tdata` + todo el cuerpo por-subfalla
del loop del dueño (antes solo cubría `add_to_response`), y los dos bloques de
comunicación MPI (envío del worker, recepción en rank 0). También se corrigió
`h5py.File(map_file, 'r+', ...)` → `'r'` en rank 0 (`run_fast` nunca escribe ahí) y se
agregó un `comm.Barrier()` final, igual que ya tiene Stage 1 (`compute_gf`).

Validado en `esmeralda` (venv/repo de prueba aislados, `testing_shakermaker`) con:
regresión numérica (código parchado vs. original, mismas funciones de Green
reutilizadas → salidas idénticas) e inyección de fallo (excepción sintética forzada
en un rank no-dueño → antes del fix el job se cuelga igual que en producción; después,
aborta limpio en segundos con traceback). Detalle completo, logs e IDs de job en
`CERNProject/10. ShakermakerModel/04_adjust_models/bug_shakermaker/`.

## Alcance / impacto

Cambio aislado a `run_fast` (Stage 2). `compute_gf` (Stage 1) y `run` (ruta legada) no
se tocaron. No cambia ningún resultado científico en el camino exitoso (comprobado por
la prueba de regresión); solo convierte un cuelgue silencioso en un abort inmediato y
diagnosticable cuando algo sí falla.

**Nota operativa, no parte del fix de código**: con 1 sola estación activa por
llamada a `run_fast` (como en esta campaña), el rank "dueño" es siempre 0 y el tiempo
total de cómputo no cambia con más ranks — los demás quedan ociosos toda la corrida.
Para campañas de pocas estaciones, pedir menos nodos/tareas en el `.sh` de Stage 2 no
cuesta tiempo de cómputo y reduce la superficie de ranks que podrían fallar.
