# Vision Line Follower (resumen en español)

> Documentación completa en inglés: [README.md](README.md). Teoría, decisiones
> de diseño y preguntas de entrevista técnica en español:
> [docs/LEARNING.md](docs/LEARNING.md).

**Vision Line Follower** es un robot de tracción diferencial **simulado** que
sigue una línea pintada en el suelo usando visión por computadora clásica
(OpenCV) -- sin hardware físico y sin aprendizaje profundo. El proyecto
simula de punta a punta:

- Un **generador de pistas 2D** procedurales (óvalo, figura en 8 con
  autointersección, bucle curvo) con curvas suaves, intersecciones pintadas,
  cambios de iluminación, ruido de textura y oclusiones -- todo controlado
  por semilla aleatoria para reproducibilidad total.
- Una **cámara simulada** montada en el robot, con modelo *pinhole* real
  (distancia focal derivada del campo de visión, montaje con altura/ángulo de
  cabeceo configurables) que renderiza la vista en perspectiva desde la pose
  real del robot mediante una única homografía por fotograma.
- Un **pipeline de visión clásico**: transformada de vista de pájaro (IPM),
  umbralización adaptativa, búsqueda por ventanas deslizantes y ajuste
  polinomial, que produce error lateral, error de rumbo y curvatura.
- Un **modelo de robot** de tracción diferencial con dinámica de motor de
  primer orden, saturación de velocidad y ruido de proceso.
- **Tres controladores comparados** de forma justa (misma señal de entrada,
  mismo programador de velocidad): PID, Pure Pursuit y Stanley (este último
  adaptado de su formulación original para vehículos tipo bicicleta a un
  robot diferencial).
- Un **banco de pruebas (benchmark)** que corre cada combinación de pista,
  controlador y nivel de ruido varias veces con semillas distintas, y mide
  error medio, error máximo, tasa de éxito (¿completó la vuelta sin salirse
  de la pista ni perder la línea?) y velocidad media -- ver
  `docs/benchmark_summary.csv` y las figuras en `docs/img/`.
- Un **GIF** del recorrido con la vista de cámara superpuesta
  (`docs/img/run_*.gif`).
- **Tests automatizados** (incluyendo imágenes sintéticas para el pipeline de
  visión) e **integración continua** en GitHub Actions (ruff, mypy, pytest en
  Python 3.11 y 3.12).

## Instalación rápida

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/run_demo.py --track oval --controller pid
python scripts/run_benchmark.py
python scripts/make_gif.py --track curvy_loop --controller stanley
```

## Resultados

Los números y gráficas exactos (generados ejecutando el código, no
estimados) están en la sección "Benchmark results" del `README.md` en
inglés y en `docs/benchmark_summary.csv`. En resumen: los tres
controladores completan la vuelta de forma confiable en condiciones
limpias y moderadas; Stanley resulta el más preciso en general, PID el
más simple y robusto, y Pure Pursuit el más sensible a curvas muy
cerradas con velocidad baja (ver la limitación documentada en
`docs/LEARNING.md`, sección 4).

## Limitaciones conocidas

Ver la sección "Known limitations & future work" del `README.md` y la
sección 4 de `docs/LEARNING.md` para el detalle completo: sin distorsión
de lente, sin dinámica de contacto rueda-suelo, sin lógica dedicada de
intersecciones, y una debilidad conocida de Pure Pursuit en el cruce
central de la pista en figura de 8 bajo cierta semilla del benchmark.

## Licencia

[MIT](LICENSE)
