# -------------------------------------------------------------
# Método de la Bisección
# -------------------------------------------------------------
# -------------------------------------------------------------
# Resumen general del programa
# - Implementa el método de la bisección para aproximar una raíz real de f(x).
# - Pide por consola: expresión de f(x) válida para Sympy, extremos a y b,
#   tolerancia y número máximo de iteraciones.
# - Entrega durante la ejecución:
#     1) Tabla por consola con iteraciones.
#     2) Archivo CSV con historial de iteraciones y errores (nombre con timestamp).
#     3) Imagen PNG con la gráfica de f(x) y la visualización de intervalos de bisección.
# - Estructura principal:
#     bisec_metodo: cómputo del método y recolección de datos.
#     guardar_csv_biseccion: mantiene la tabla en CSV.
#     graficar_funcion_y_biseccion: gráfico con f(x) e intervalos.
#     bloque main: gestión de entradas y de pasos.
# -------------------------------------------------------------

import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime


def bisec_metodo(func_str, a, b, tol, max_iter):
    """
    Aplica el método de la bisección a f en [a,b].
    Devuelve: (raiz_aprox, tabla_resultados, historial_intervalos)
      - raiz_aprox: último c_k
      - tabla_resultados: lista de filas [i, a_i, b_i, c_i, f(c_i), error]
      - historial_intervalos: lista de tuplas (a_i, b_i, c_i) para graficar
    """
    # Variable simbólica y función numérica
    # Sympify interpreta texto como expresión de Sympy.
    x = sp.symbols('x')
    f_expr = sp.sympify(func_str)
    f = sp.lambdify(x, f_expr, 'numpy')

    # Evaluaciones iniciales en los extremos del intervalo.
    # Para verificar el cambio de signo previo al método.
    fa = f(a)
    fb = f(b)

    # Estructuras para acumular resultados y para graficar.
    tabla_resultados = []
    historial_intervalos = []

    # Comprobaciones iniciales
    # Si f(a) o f(b) no son numéricos válidos se aborta.
    if np.isnan(fa) or np.isnan(fb):
        print("Error: f(a) o f(b) no son evaluables (NaN).")
        return None, tabla_resultados, historial_intervalos

    # Casos frontera: raíz exacta en uno de los extremos.
    # Se devuelve con una fila informativa y error 0.
    if fa == 0:
        print("Se encontró raíz exacta en a.")
        tabla_resultados.append([0, a, b, a, f(a), 0.0])
        historial_intervalos.append((a, b, a))
        return a, tabla_resultados, historial_intervalos

    if fb == 0:
        print("Se encontró raíz exacta en b.")
        tabla_resultados.append([0, a, b, b, f(b), 0.0])
        historial_intervalos.append((a, b, b))
        return b, tabla_resultados, historial_intervalos

    # Condición necesaria del teorema del valor intermedio:
    # si f(a) y f(b) tienen el mismo signo no se garantiza raíz en [a,b].
    if fa * fb > 0:
        print("Error: f(a) y f(b) tienen el mismo signo. No se garantiza raíz en [a,b].")
        return None, tabla_resultados, historial_intervalos

    # Cabecera de tabla para la salida por consola.
    # Anchos fijos para lectura alineada.
    print(f"\n{'Iteración':^10} | {'a_i':^18} | {'b_i':^18} | {'c_i':^18} | {'f(c_i)':^18} | {'Error':^12}")
    print("-" * 108)

    # Bucle principal de iteración de la bisección.
    # En cada paso se parte el intervalo por el punto medio c.
    for i in range(1, max_iter + 1):
        #  Punto medio del intervalo actual.
        c = 0.5 * (a + b)
        fc = f(c)

        # Definición del error en bisección: Semilongitud del intervalo.
        error = 0.5 * (b - a)

        # Registro de resultados para CSV y gráfica.
        print(f"{i:^10} | {a:^18.10f} | {b:^18.10f} | {c:^18.10f} | {fc:^18.10e} | {error:^12.3e}")

        # Guardar en estructuras
        tabla_resultados.append([i, a, b, c, fc, error])
        historial_intervalos.append((a, b, c))

        # Criterios de parada:
        # 1) valor de función cero exacto,
        # 2) error menor que tolerancia.
        if abs(fc) == 0 or error < tol:
            print("\nConvergencia alcanzada.")
            return c, tabla_resultados, historial_intervalos

        # Selección del subintervalo con cambio de signo.
        # Si f(a) y f(c) tienen signo opuesto, la raíz está en [a,c];
        # en caso contrario está en [c,b].
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    # Si se agotan las iteraciones sin cumplir el criterio:
    print("\nNo se alcanzó la convergencia con el número máximo de iteraciones.")
    # Se devuelve el último c y los historiales recopilados.
    return c, tabla_resultados, historial_intervalos


# -------------------------------------------------------------
# Guardar CSV
# -------------------------------------------------------------
def guardar_csv_biseccion(tabla, archivo_csv):
    # Escritura de la tabla de iteraciones a CSV.
    with open(archivo_csv, mode='w', newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["Iteración", "a_i", "b_i", "c_i", "f(c_i)", "Error"])
        for fila in tabla:
            i, ai, bi, ci, fci, error = fila
            writer.writerow([
                i-1,  # Iteración empieza en 0 en CSV
                f"{ai:.8f}",
                f"{bi:.8f}",
                f"{ci:.8f}",
                f"{fci:.8e}",
                f"{error:.8e}"
            ])


# -------------------------------------------------------------
# Gráfico de función + visualización de iteraciones (bisección)
# -------------------------------------------------------------
def graficar_funcion_y_biseccion(f_expr, historial_intervalos, a_ini, b_ini, archivo_salida):
    # Preparación de función numérica.
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr, 'numpy')

    # Malla de puntos para la función
    x_plot = np.linspace(a_ini, b_ini, 600)
    y_plot = f(x_plot)

    plt.figure(figsize=(10, 6))
    # Línea horizontal en y=0 para referencia visual.
    plt.axhline(0, color='gray', linestyle='--')

    # Etiqueta de leyenda generada desde LaTeX de Sympy.
    # formula_latex se usa como texto de leyenda.
    formula_latex = r"$" + sp.latex(f_expr) + r"$"
    plt.plot(x_plot, y_plot, label=formula_latex, linewidth=2)

    # Dibujo de los intervalos en cada iteración:
    # líneas verticales en a_i y b_i y el punto medio c_i.
    for k, (ai, bi, ci) in enumerate(historial_intervalos, start=1):
        # Líneas verticales del intervalo en esta iteración
        plt.plot([ai, ai], [0, f(ai)], linestyle='--', color='red', alpha=0.6)
        plt.plot([bi, bi], [0, f(bi)], linestyle='--', color='red', alpha=0.6)
        # Punto medio
        plt.plot(ci, f(ci), 'ro', markersize=4, alpha=0.8)

    # Último punto (aproximación final) en verde
    if historial_intervalos:
        _, _, c_final = historial_intervalos[-1]
        plt.plot(c_final, f(c_final), 'go', label='Aproximación final')

    plt.title("Método de la Bisección")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig(archivo_salida, dpi=150, bbox_inches='tight')
    plt.close()


# -------------------------------------------------------------
# Programa principal
# -------------------------------------------------------------
if __name__ == "__main__":
    # Timestamp con formato YYYYMMDD_HHMMSS para nombres únicos de archivos.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Método de la Bisección\n")

    # Entrada de usuario.
    # funcion se captura como texto interpretable por Sympy.
    funcion = input("Introduce la función f(x): ")

    # Intervalo [a,b] permitiendo constantes simbólicas como pi.
    # sympify convierte el texto a expresión y luego se hace cast a float.
    a = float(sp.sympify(input("Extremo izquierdo del intervalo a (puedes usar pi): ")))
    b = float(sp.sympify(input("Extremo derecho del intervalo b (puedes usar pi): ")))

    # Parámetros
    tol = float(input("Tolerancia (por ejemplo 1e-6): "))
    max_iter = int(input("Número máximo de iteraciones: "))

    # Asegurar a <= b
    if a > b:
        a, b = b, a

    # Ejecución del método y recogida de resultados.
    raiz, tabla, historial = bisec_metodo(funcion, a, b, tol, max_iter)

    # Guardar CSV si hubo iteraciones (o se detectó raíz en extremo)
    if tabla:
        csv_filename = f"resultados_biseccion_{timestamp}.csv"
        guardar_csv_biseccion(tabla, csv_filename)
        print(f"Tabla de resultados guardada en '{csv_filename}'.")

    # Generar gráfico si hubo historial de intervalos
    if historial:
        img_filename = f"grafico_biseccion_{timestamp}.png"
        graficar_funcion_y_biseccion(sp.sympify(funcion), historial, a, b, img_filename)
        print(f"Gráfico guardado en '{img_filename}'.")

    # Mostrar raíz aproximada si disponible
    if raiz is not None:
        print(f"\nRaíz aproximada: {raiz}")
