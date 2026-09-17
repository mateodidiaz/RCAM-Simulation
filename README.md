# RCAM — simulación de dinámica de vuelo

Implementación en Python de un modelo no lineal RCAM (Research Civil Aircraft Model), simulaciones con entradas de control y fallo de motor, y búsqueda de condiciones de equilibrio mediante Particle Swarm Optimization (PSO).

## Contenido

| Archivo | Función |
|---|---|
| `rcam_model.py` | Calcula las derivadas de nueve estados a partir de cinco entradas de control. |
| `simulate_p2_3_4A.py` | Simula vuelo con controles constantes, pulso de alerón o fallo de motor; genera gráficas y una animación GIF. |
| `simulate_p4B_pso.py` | Busca condiciones de trim con un motor apagado y muestra la respuesta del modelo con la solución obtenida. |

Estados: `X = [u, v, w, p, q, r, phi, theta, psi]`. Entradas: `U = [alerón, estabilizador, timón, empuje_1, empuje_2]`. Las velocidades se expresan en m/s; los ángulos, en radianes; y las velocidades angulares, en rad/s. Los estados se integran con RK4; la posición NED se integra por Euler en el script de simulación.

## Instalación

Con Python 3 y pip disponibles, desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En Linux/macOS usa `.venv/bin/python` en lugar de `.\.venv\Scripts\python.exe`.

## Simulaciones

```powershell
.\.venv\Scripts\python.exe simulate_p2_3_4A.py
```

Edita `SIMULATION_TO_RUN` al principio del script:

- `2`: controles constantes.
- `3`: pulso de alerón de 5 grados entre los segundos 30 y 32.
- `4`: motor apagado desde el inicio. `FAILED_ENGINE` selecciona el motor 1 o 2.

`TF`, `DT_OUT` y `DT_INT` controlan la duración y los pasos temporales. La configuración inicial selecciona el escenario 4 y el fallo del motor 2. Las ventanas de Matplotlib aparecen en secuencia; ciérralas para continuar hasta la exportación del GIF.

## Optimización PSO

```powershell
.\.venv\Scripts\python.exe simulate_p4B_pso.py
```

El script busca una condición cercana a 78 m/s con un motor apagado, usando 45 partículas y 120 iteraciones. `FAILED_ENGINE` se configura de forma independiente en este archivo; su valor inicial es 1. Muestra el coste, los estados, los controles, las derivadas, la convergencia y la respuesta posterior.

La búsqueda es estocástica y no fija una semilla. Un coste pequeño no demuestra equilibrio exacto, estabilidad ni un mínimo global: deben evaluarse las derivadas y la trayectoria resultante. Las dependencias no están fijadas a versiones específicas.

## Alcance de la publicación

Se distribuyen los scripts y las instrucciones para ejecutarlos. El informe académico local queda excluido porque contiene datos personales. Las gráficas y animaciones se generan al ejecutar los programas y no se incluyen en el repositorio. Este proyecto es independiente de cualquier integración con APIs financieras.
