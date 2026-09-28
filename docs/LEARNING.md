# Guía de aprendizaje: Vision Line Follower

Este documento explica, en español, la teoría detrás de cada componente del proyecto,
las decisiones de diseño tomadas (y las alternativas que se descartaron y por qué),
sus limitaciones conocidas, y cierra con diez preguntas de entrevista técnica —con
respuesta— basadas directamente en el código de este repositorio.

No sustituye la lectura del código fuente (que incluye docstrings extensos), pero
sirve como mapa de navegación y como justificación escrita de las decisiones de
ingeniería que no son obvias con solo leer una función aislada.

## Índice

1. [Visión general del sistema](#1-visión-general-del-sistema)
2. [Teoría paso a paso](#2-teoría-paso-a-paso)
   - [2.1 Generación de pistas](#21-generación-de-pistas)
   - [2.2 Modelo de cámara y homografías](#22-modelo-de-cámara-y-homografías)
   - [2.3 Pipeline de visión clásico](#23-pipeline-de-visión-clásico)
   - [2.4 Modelo del robot](#24-modelo-del-robot)
   - [2.5 Controladores](#25-controladores)
   - [2.6 Bucle de simulación y métricas](#26-bucle-de-simulación-y-métricas)
3. [Decisiones de diseño y alternativas descartadas](#3-decisiones-de-diseño-y-alternativas-descartadas)
4. [Limitaciones conocidas y trabajo futuro](#4-limitaciones-conocidas-y-trabajo-futuro)
5. [Diez preguntas de entrevista técnica](#5-diez-preguntas-de-entrevista-técnica)

---

## 1. Visión general del sistema

El proyecto simula, de punta a punta, un robot de tracción diferencial que sigue una
línea pintada en el suelo usando **únicamente visión por computadora clásica**
(sin aprendizaje profundo): una cámara simulada monta en el robot, ve la línea con
perspectiva, y un pipeline de OpenCV la transforma en dos números —error lateral y
error de rumbo— más una estimación de curvatura, que alimentan a uno de tres
controladores (PID, Pure Pursuit o Stanley) para decidir cuánto girar.

Todo el sistema es determinista dado un seed: la geometría de la pista, la textura
del suelo, el ruido de la cámara y el ruido del motor dependen de semillas
explícitas, así que cualquier corrida es 100 % reproducible.

## 2. Teoría paso a paso

### 2.1 Generación de pistas

Cada pista nace de un puñado de **puntos de control** perturbados aleatoriamente
(según el seed) y dispuestos a mano en tres formas base: óvalo, figura en 8 (con
autointersección) y bucle curvo (radio variable). Esos puntos se interpolan con un
**B-spline cúbico periódico** (`scipy.interpolate.splprep(..., per=True)`), se
muestrean densamente (4000 puntos) y luego se **re-muestrean a espaciado uniforme
por longitud de arco** interpolando linealmente sobre la longitud acumulada.

Con el muestreo uniforme, el rumbo y la curvatura se calculan con diferencias
centrales simples:

```
heading[i]   = atan2(y[i+1] - y[i-1], x[i+1] - x[i-1])
curvature[i] = dtheta/ds   (con "dtheta" envuelto a (-pi, pi] para evitar
                            saltos falsos de 2*pi en la costura de la curva cerrada)
```

Ese envolvimiento angular (`wrap`) es crítico: como la pista es cerrada, un
`np.unwrap` ingenuo sobre todo el arreglo introduce un salto artificial de ~2π en el
punto donde el arreglo "da la vuelta", disparando valores de curvatura absurdos
justo en un punto arbitrario de la pista. La solución (ver
`track/generator.py::generate_track`) es calcular la diferencia de rumbo entre
vecinos con una envoltura local, no con un desenvolvimiento global.

### 2.2 Modelo de cámara y homografías

La cámara es un modelo *pinhole* ideal, montada mirando hacia adelante y hacia
abajo (ángulo de cabeceo configurable). Como todos los puntos de interés están en
el plano del suelo (`z = 0` en el marco del robot), la proyección de un punto del
suelo a un píxel de cámara es una **homografía 3×3** (no una transformación
proyectiva general de 3D a 2D, que necesitaría más parámetros).

Esto se explota dos veces en el proyecto:

1. **Renderizado de la cámara** (`sim/camera.py`): en vez de renderizar en 3D, se
   compone una única homografía por fotograma que va de píxel del "mapa del mundo"
   pre-renderizado directamente a píxel de cámara:
   `H_total = H_suelo→cámara · T_mundo→local · S_píxel_mundo→metro_mundo`. Las tres
   son matrices 3×3 homogéneas (dos afines, una proyectiva), así que su producto es
   una sola homografía que `cv2.warpPerspective` aplica en un solo paso. El costo
   de renderizar no depende del tamaño del mapa del mundo, solo de la resolución de
   salida.
2. **Transformada de vista de pájaro / IPM** (`vision/birdseye.py`): se toman
   cuatro esquinas de un rectángulo del suelo justo frente al robot (en el marco
   local del robot, sin necesidad de conocer su pose en el mundo), se proyectan con
   la misma homografía de cámara, y `cv2.getPerspectiveTransform` calcula la
   homografía inversa que "endereza" esa vista en perspectiva a una vista cenital.
   Esto imita exactamente cómo un robot real calibraría su propia IPM una sola vez,
   usando su calibración de cámara conocida, sin necesitar el mapa del mundo.

### 2.3 Pipeline de visión clásico

El pipeline reproduce, para una sola línea, la técnica clásica de detección de
carriles ("advanced lane finding"):

1. **Vista de pájaro (IPM)** — descrita arriba.
2. **Umbralización adaptativa** (`vision/threshold.py`): en vez de un umbral fijo
   (`img > 127`), se usa `cv2.adaptiveThreshold` con una ventana local gaussiana.
   Esto hace que el sistema tolere gradientes de iluminación y viñeteo simulados
   sobre la pista: un umbral global fallaría en cuanto una esquina de la imagen
   estuviera más oscura que otra por razones ajenas a la línea.
3. **Búsqueda por ventanas deslizantes** (`vision/sliding_window.py`): se parte de
   un histograma de columnas en la mitad inferior (más cercana al robot) de la
   máscara binaria, se toma el pico como punto de partida, y se apilan ventanas
   hacia arriba (más lejos), recentrando cada ventana sobre la media de los píxeles
   activos que contiene. El resultado es una nube de puntos `(fila, columna)` que
   pertenecen a la línea.
4. **Ajuste polinomial** (`vision/fit.py`): se ajusta un polinomio cuadrático
   `lateral = c2·adelante² + c1·adelante + c0` por mínimos cuadrados
   (`np.polyfit`). De ese polinomio se leen directamente:
   - **Error lateral** = valor del polinomio en el punto más cercano del ROI.
   - **Error de rumbo** = `atan(pendiente)` en ese mismo punto.
   - **Curvatura** = `f'' / (1 + f'²)^1.5` (fórmula estándar de curvatura de una
     curva `y = f(x)`).

Un detalle de robustez añadido: la búsqueda por ventanas deslizantes recuerda la
columna base de la ventana inferior del fotograma anterior (`prior_x_px`) y, si se
le da, restringe el histograma inicial a una vecindad alrededor de ese valor en vez
de tomar el máximo global. Esto evita que, en una intersección en X (como el cruce
central de la figura en 8), el buscador "salte" a la rama equivocada solo porque
tiene más píxeles blancos en ese fotograma.

### 2.4 Modelo del robot

El robot es un modelo de **tracción diferencial** con dinámica de motor de primer
orden: la velocidad real de cada rueda no salta instantáneamente al valor
comandado, sino que se relaja exponencialmente
(`v += (1 - exp(-dt/tau)) · (v_comandada - v)`), se satura a un máximo físico, y se
le suma ruido gaussiano de proceso (deslizamiento de rueda / jitter de PWM). La
integración de pose usa la solución exacta de arco circular para `(v, omega)`
constantes durante el paso, en vez de una integración de Euler de primer orden, lo
que reduce el error de discretización en curvas cerradas con pasos de tiempo
relativamente grandes.

### 2.5 Controladores

Los tres controladores comparten la misma interfaz: reciben `(error_lateral,
error_de_rumbo, curvatura, velocidad, dt)` y devuelven una velocidad angular
`omega`. La velocidad lineal **no** la decide el controlador: la decide un
programador de velocidad común (`SpeedSchedule`) en función de la curvatura
estimada, para que la comparación entre controladores no esté contaminada por
diferencias en la estrategia de velocidad.

- **PID** (`control/pid.py`): controla un error combinado
  `e = error_lateral + peso_rumbo · error_de_rumbo` con las tres ganancias
  clásicas, anti-windup por recorte del término integral, y derivada por
  diferencia finita.
- **Pure Pursuit** (`control/pure_pursuit.py`): el Pure Pursuit clásico ajusta un
  arco circular desde el eje trasero hasta un punto de referencia (*lookahead*) a
  distancia fija sobre la trayectoria. Aquí no se dispone de la trayectoria
  completa, solo de los tres escalares del pipeline, así que se **reconstruye**
  un modelo cuadrático local de la línea (expansión de Taylor alrededor del punto
  donde el pipeline evalúa el error) y se evalúa en el punto de lookahead. La
  curvatura de Pure Pursuit está acotada por `2 / lookahead`, así que el
  *lookahead* se reduce con la velocidad mínima para poder alcanzar curvaturas
  altas en curvas cerradas.
- **Stanley** (`control/stanley.py`): el Stanley original (Thrun et al., 2006,
  Stanford Racing Team) calcula un ángulo de dirección para un vehículo tipo
  bicicleta: `delta = error_rumbo + atan2(k·error_lateral, v + k_soft)`. Como un
  robot diferencial no tiene una rueda dirigida, ese ángulo se convierte en
  velocidad angular con la misma relación cinemática que usaría un modelo de
  bicicleta (`omega = v·tan(delta)/L_efectivo`), con `L_efectivo` como parámetro
  de ajuste — una adaptación explícita, no el Stanley de libro de texto.

### 2.6 Bucle de simulación y métricas

En cada paso (`sim/world.py::run_simulation`): se renderiza la cámara, se corre el
pipeline de visión, se aplica un filtro EMA temporal a la estimación (atenúa
lecturas ruidosas puntuales, por ejemplo cerca de una intersección), se calcula la
velocidad objetivo por curvatura, se pide `omega` al controlador, se convierte a
velocidades de rueda y se avanza el robot. El **error de referencia (cross-track
error)** que se reporta en las métricas **no** es el error estimado por visión: es
la distancia con signo desde la pose real del robot hasta el punto más cercano del
centro de la pista (la verdad de terreno, ground truth), calculada de forma
completamente independiente del pipeline de visión. Esto evita que un sistema de
visión "optimista" (que subestima su propio error) parezca mejor de lo que es en
el benchmark.

## 3. Decisiones de diseño y alternativas descartadas

**Homografías analíticas en vez de calibración por correspondencia de puntos.**
Se podría haber calibrado la IPM con `cv2.findHomography` sobre puntos de
correspondencia manuales, como en muchos tutoriales. Se optó por derivar la
homografía **analíticamente** desde los parámetros intrínsecos/extrínsecos de la
cámara porque (a) es exacta, sin error de ajuste; (b) permite recalcular la
calibración instantáneamente si cambian los parámetros de montaje de la cámara; y
(c) es más representativo de cómo se calibraría un robot real con su hoja de datos
de cámara conocida.

**B-splines periódicos en vez de generación de pistas basada en tiles/grid.**
Un generador basado en cuadrícula (como muchos simuladores de robots móviles)
produce esquinas artificiales de 90°. Se prefirió una curva suave y continua porque
el objetivo es *seguimiento de línea con visión*, no navegación discreta, y las
pistas de competencias reales de seguidores de línea son curvas suaves con
curvatura variable, no cuadrículas.

**Pipeline de visión clásico en vez de segmentación con red neuronal.**
Una CNN de segmentación semántica sería más robusta a condiciones extremas, pero
el enunciado del proyecto pide explícitamente el pipeline clásico (transformada de
vista de pájaro, umbralización adaptativa, ventanas deslizantes, ajuste
polinomial), que además es más barato computacionalmente, totalmente interpretable
paso a paso, y no requiere datos de entrenamiento — apropiado para un robot
embebido de bajo costo.

**Reconstrucción local (Taylor) para Pure Pursuit en vez de pasar la trayectoria
completa por la interfaz del controlador.** Se decidió mantener una interfaz común
de tres escalares para los tres controladores (fiel a lo que pide el enunciado:
"error lateral, error de rumbo y curvatura" como salidas del pipeline). El costo es
que Pure Pursuit debe reconstruir un modelo local de la línea para evaluarlo en su
punto de *lookahead*, lo cual es exacto solo cerca del punto de referencia del
pipeline y se degrada al extrapolar lejos en curvas muy cerradas — ver limitaciones.

**Filtro EMA temporal aplicado a todos los controladores por igual.** En vez de
intentar arreglar cada controlador individualmente para tolerar lecturas de visión
ruidosas puntuales (por ejemplo en un cruce en X), se añadió un filtro de media
móvil exponencial **antes** de que el error llegue a cualquier controlador. Es una
técnica estándar en sistemas de visión embebidos reales, y mantiene la comparación
justa: los tres controladores ven exactamente la misma señal filtrada.

**Velocidad programada por curvatura, común a los tres controladores.** Si cada
controlador decidiera también su propia velocidad, sería imposible saber si las
diferencias de error en el benchmark vienen del controlador de dirección o de la
estrategia de velocidad. Separar ambas decisiones aísla la variable que el proyecto
quiere comparar.

**Integración de arco exacto en vez de Euler simple.** Con pasos de 20 ms y
velocidades angulares de varios rad/s en curvas cerradas, Euler de primer orden
acumula error de forma visible en unos pocos cientos de pasos. La solución de arco
circular cerrada (`sim/robot.py::DifferentialDriveRobot.step`) es igual de barata
de calcular y elimina ese sesgo sistemático.

**Ground truth de error independiente del error estimado por visión.** Ver
sección 2.6: es la única forma honesta de comparar controladores sin premiar a un
pipeline de visión que subestima su propio error.

## 4. Limitaciones conocidas y trabajo futuro

Documentadas explícitamente, en vez de omitidas en silencio, como pide el alcance
del proyecto:

- **Pure Pursuit en la autointersección de la figura en 8.** En el benchmark
  completo (ver `docs/benchmark_summary.csv`), Pure Pursuit completa el óvalo y
  el bucle curvo de forma perfecta (100 % de éxito en las semillas evaluadas)
  pero solo el 58-67 % de las corridas en la pista de figura en 8, y casi todos
  los fallos ocurren en el cruce central: el error de rumbo estimado se acerca a
  ±90° durante varios fotogramas seguidos (el robot gira sobre sí mismo con muy
  poca velocidad de avance) y la ley de control geométrica no genera suficiente
  curvatura de giro para converger a tiempo antes de que la línea salga del
  campo de visión. Se intentó mitigar de varias formas: recorte de la
  extrapolación de Taylor para evitar el desborde numérico de la tangente,
  anclaje temporal de la ventana deslizante, y un término de reducción de
  *lookahead* proporcional a la curvatura — este último sí ayuda en el cruce de
  la figura en 8 (ver `PurePursuitConfig.lookahead_curvature_gain`), pero un
  valor de ganancia lo bastante alto para arreglar ese cruce **rompe** el cierre
  de vuelta del óvalo y del bucle curvo (el robot se sale de pista justo antes
  de completar la vuelta, alrededor del 95 % de recorrido), así que se descartó
  y se mantuvo la ganancia en 0. Es una limitación conocida de Pure Pursuit
  frente a curvas muy cerradas combinadas con velocidad mínima baja, no un error
  de implementación oculto — y precisamente el tipo de hallazgo que un benchmark
  honesto debe sacar a la luz en vez de ocultar ajustando el banco de pruebas.
  Trabajo futuro: *lookahead* adaptativo específico por tramo de pista (no solo
  por curvatura puntual), o pasar la trayectoria completa (no solo el punto de
  evaluación) al controlador.
- **Extrapolación local para Pure Pursuit.** El modelo cuadrático reconstruido por
  Taylor solo es fiel cerca del punto donde el pipeline evalúa el error; para
  *lookahead* distances mucho mayores que el ROI de visión, la extrapolación pierde
  precisión. Mitigado con un recorte del error de rumbo usado en la expansión, pero
  no eliminado.
- **Distorsión de lente no simulada.** La cámara es un pinhole ideal sin distorsión
  radial/tangencial. Cámaras baratas reales tienen distorsión significativa,
  especialmente en los bordes de gran angular.
- **Sin fricción de neumático ni deslizamiento lateral.** El modelo de robot
  integra cinemática pura (con dinámica de motor de primer orden), no una dinámica
  de contacto rueda-suelo con deslizamiento lateral.
- **Detección de intersecciones sin lógica dedicada.** Las intersecciones
  perpendiculares pintadas sobre la pista son visuales/geométricas, pero ningún
  controlador tiene lógica explícita de "esto es un cruce, decide qué rama tomar":
  el sistema simplemente confía en que la línea principal siga siendo la de mayor
  continuidad local. Para un seguidor de línea con lógica de intersección real
  (parar, contar cruces, girar en un cruce específico) haría falta un
  planificador de más alto nivel.
- **Ruido de textura como aproximación de condiciones reales.** El ruido de
  iluminación, viñeteo y textura del piso son aproximaciones razonables pero
  sintéticas; no sustituyen un dataset de imágenes reales.

## 5. Diez preguntas de entrevista técnica

**1. ¿Qué es una homografía y por qué la transformación cámara→suelo es una
homografía y no una proyección 3D genérica?**

Una homografía es una transformación proyectiva 2D↔2D representable como una
matriz 3×3 que actúa sobre coordenadas homogéneas. La proyección de un punto 3D
arbitrario a la imagen de una cámara *no* es, en general, invertible ni
representable con una matriz 3×3 (se pierde la profundidad). Pero si todos los
puntos de interés están restringidos a un plano (aquí, el plano del suelo,
`z = 0`), la composición de la restricción al plano con la proyección de cámara
sí es una transformación 2D↔2D completa, y por tanto una homografía. Por eso el
proyecto puede calcular `H = K · [r1 r2 t]` (usando solo las dos primeras columnas
de la rotación) en vez de la matriz de proyección 3×4 completa.

**2. ¿Cuál es la diferencia práctica entre Pure Pursuit y Stanley?**

Pure Pursuit es puramente geométrico: ajusta un arco circular hasta un punto de
la trayectoria a una distancia fija (*lookahead*) y no reacciona con la misma
fuerza al error lateral cuando ese error es pequeño (tiende a "cortar" curvas).
Stanley controla explícitamente **dos** términos a la vez —error de rumbo y error
lateral normalizado por velocidad— evaluados en el eje delantero del vehículo, lo
que en general converge más agresivamente a error lateral cero en línea recta,
pero puede ser más sensible al ruido de la medición de error lateral a baja
velocidad (el término `atan2(k·e, v)` se dispara si `v` es pequeño, de ahí la
constante de suavizado `k_soft`).

**3. ¿Por qué el controlador PID de este proyecto no controla el error lateral y
el error de rumbo por separado, con dos lazos PID independientes?**

Dos lazos PID acoplados (ambos afectan la misma salida, `omega`) son más difíciles
de sintonizar de forma estable: hay que evitar que se contradigan. La solución
adoptada —un único error combinado `lateral + peso · rumbo`— es una simplificación
práctica común en controladores de línea embebidos de bajo costo, a cambio de un
grado de libertad menos en la sintonización.

**4. ¿Qué es el "anti-windup" en un PID y por qué se implementa aquí con un
recorte (clamp) del término integral?**

El "windup" ocurre cuando el error persiste (por ejemplo mientras la salida está
saturada) y el término integral crece sin límite, causando un sobreimpulso grande
cuando el error finalmente cambia de signo. El recorte directo del acumulador
integral (`PIDConfig.integral_limit`) es la forma más simple de anti-windup:
acota el término integral a un rango razonable independientemente de cuánto tiempo
persista el error.

**5. ¿Por qué el *lookahead* de Pure Pursuit se reduce cuando la curvatura
estimada es alta, en vez de mantenerse fijo?**

Porque la curvatura máxima que Pure Pursuit puede comandar está geométricamente
acotada por `kappa_max = 2 / lookahead`. Si el *lookahead* es grande (apropiado
para tramos rectos, donde da estabilidad), el controlador nunca podrá generar la
curvatura que exige una curva cerrada, sin importar cuán grande sea el error
medido. Reducir el *lookahead* en curvas (`lookahead_curvature_gain`) sube ese
techo justo cuando se necesita.

**6. ¿Cómo se calcula la curvatura de una curva descrita por un polinomio
`y = f(x)`, y por qué el proyecto usa un polinomio de grado 2?**

La fórmula general es `kappa = f'' / (1 + f'^2)^{3/2}`. Con un polinomio
cuadrático (`f'' `constante), la curvatura solo depende de la posición a través
del término `f'`, lo cual es una aproximación de curvatura localmente constante —
razonable en el rango corto (unos 30-40 cm) que cubre el ROI de la cámara. Un
grado mayor (cúbico) permitiría capturar cambios de curvatura dentro del ROI, a
costa de mayor sensibilidad al ruido de los píxeles detectados (más parámetros
ajustados con la misma cantidad de puntos).

**7. ¿Por qué usar umbralización *adaptativa* en vez de un umbral fijo tipo
Otsu global?**

Un umbral global asume iluminación uniforme en toda la imagen. En este proyecto la
pista simula gradientes de iluminación y viñeteo de lente deliberadamente, así que
una esquina de la imagen puede ser sistemáticamente más oscura que otra sin que
eso tenga nada que ver con la línea. `cv2.adaptiveThreshold` calcula un umbral
distinto para cada vecindad local de píxeles (comparando cada píxel contra la
media/gaussiana de su entorno), lo que lo hace insensible a esas variaciones
lentas y solo sensible al contraste local línea-vs-piso.

**8. ¿Qué ventajas ofrece simular todo el sistema (pista, cámara, robot) en vez
de probar directamente sobre hardware real?**

Reproducibilidad total (mismo seed ⇒ mismo resultado, indispensable para
comparar controladores de forma justa), posibilidad de barrer sistemáticamente
niveles de ruido/iluminación/oclusión que serían tediosos o imposibles de
controlar con precisión en el mundo físico, velocidad de iteración (no hay que
recargar baterías ni recalibrar una cámara física entre pruebas), y seguridad
(nada se choca ni se rompe mientras se ajustan ganancias de control agresivas).

**9. En un sistema con aleatoriedad (ruido de cámara, ruido de motor, textura del
piso), ¿cómo se garantiza que los resultados sean reproducibles?**

Cada fuente de aleatoriedad usa su **propio** generador `numpy.random.Generator`
sembrado explícitamente (`np.random.default_rng(seed)`), nunca el estado global de
NumPy. El proyecto separa semillas por responsabilidad: una para la geometría de
la pista, otra para la textura/oclusiones del renderizado, otra para el ruido de
píxeles de cámara, otra para el ruido de proceso del robot. Fijar todas las
semillas reproduce exactamente la misma corrida byte a byte; cambiar solo una
(por ejemplo, la semilla de la corrida en el benchmark) produce una muestra
estadísticamente independiente sin afectar la pista o el renderizado de fondo.

**10. ¿Qué problema resuelve la búsqueda por ventanas deslizantes que no resuelve
simplemente tomar todos los píxeles blancos de la máscara binaria?**

Tomar todos los píxeles blancos de la imagen completa mezclaría, sin distinción,
píxeles que pertenecen a la línea principal con píxeles de una intersección
perpendicular, una oclusión, o ruido residual del umbralado. La búsqueda por
ventanas deslizantes impone una restricción estructural: solo cuenta como parte de
la línea aquello que es alcanzable por una secuencia continua de ventanas
recentradas empezando desde el punto más cercano al robot, lo que en la práctica
filtra la mayoría de detecciones espurias que no forman una curva continua
plausible partiendo desde la posición actual del robot.
