"""
Programa: Método de Newton-Raphson con salida a CSV y gráfico de iteraciones

Qué hace:
    - Pide al usuario una función f(x), un intervalo [a, b], un punto inicial x0,
      una tolerancia y un máximo de iteraciones.
    - Ejecuta el método de Newton-Raphson para aproximar una raíz de f(x).
    - Muestra una tabla por consola con las iteraciones.
    - Guarda la tabla en un archivo CSV con timestamp en el nombre.
    - Genera un gráfico con la curva de f(x) y las 'escaleras' de Newton.

Entradas (vía input):
    - funcion: string con una expresión simbólica válida para Sympy (p.ej. 'x**3 - 2*x - 5', 'sin(x) - x/2)'.
    - a, b: extremos del intervalo (admite constantes simbólicas como 'pi').
    - x0: punto inicial (admite 'pi', etc.).
    - tol: tolerancia para el criterio de parada (p.ej. 1e-6).
    - max_iter: máximo de iteraciones del método.

Salidas:
    - Impresión por consola de la tabla de iteraciones (Iteración, x1, f(x1), Error).
    - Fichero CSV con los resultados (nombre con timestamp).
    - Imagen PNG con la función y las iteraciones (nombre con timestamp).
    - Mensaje con la raíz aproximada si se cumple el criterio de convergencia.

Notas:
    - Newton-Raphson requiere f'(x0) no 0 y un x0 razonable.
    - Newton no 'respeta' el intervalo en general; aquí se comprueba y se detiene
      si la iteración sale de [a, b] (decisión de diseño para seguridad).
    - Se usa sympify para permitir constantes simbólicas en la entrada y lambdify para evaluar con NumPy.
"""
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime

# Función que aplica el método de Newton-Raphson
def newton_metodo(func_str, x0, a, b, tol, max_iter):
    """
    Ejecuta el método de Newton-Raphson sobre la función dada como string.

    Parámetros:
        func_str (str): expresión de f(x) interpretable por Sympy (p.ej., 'x**2 - 2').
        x0 (float): punto inicial.
        a, b (float): extremos del intervalo de seguridad para validar la iteración.
        tol (float): tolerancia para el criterio |x_{n+1} - x_n| < tol.
        max_iter (int): máximo número de iteraciones permitidas.

    Devuelve:
        (raiz, tabla_resultados, x_val)
            raiz (float | None): aproximación si converge, None en caso contrario.
            tabla_resultados (list[list]): filas [i, x1, f(x1), error] por iteración.
            x_val (list[float]): secuencia de iterados (para graficar 'escaleras').
    """
    # Definimos la variable simbólica
    x = sp.symbols('x')

    # Declaramos la función y su derivada como expresiones simbólicas
    # Nota: sympify permite que el usuario teclee 'pi', 'sin(x)', etc.
    f_expr = sp.sympify(func_str)
    f_prim_expr = sp.diff(f_expr, x)
    # f_2prim_expr = sp.diff(f_prim_expr, x)

    # Creamos funciones numéricas evaluables con numpy
    # lambdify traduce la expresión simbólica a una función que acepta floats/arrays.
    f = sp.lambdify(x, f_expr, 'numpy')
    f_prim = sp.lambdify(x, f_prim_expr, 'numpy')
    # f_2prim = sp.lambdify(x, f_2prim_expr, 'numpy')

    # Almacenamos resultados de cada iteración
    tabla_resultados = []
    x_val = [x0]  # Lista para graficar el recorrido de iteraciones

    # Imprimimos el encabezado de la tabla en consola
    print(f"\n{'Iteración':^10} | {'x1':^20} | {'f(x1)':^20} | {'Error':^15}")
    print("-" * 75)

    for i in range(1, max_iter + 1):
        f_x0 = f(x0)
        f_prim_x0 = f_prim(x0)
        # f_2prim_x0 = f_2prim(x0)

        # Evitamos división por cero
        if f_prim_x0 == 0:
            print("Derivada nula. Método detenido.")
            return None, tabla_resultados, x_val

        # Cálculo de siguiente punto y error
        x1 = x0 - f_x0 / f_prim_x0
        f_x1 = f(x1)
        error = abs(x1 - x0)

        # Mostrar en consola
        print(f"{i:^10} | {x1:^20.10f} | {f_x1:^20.10e} | {error:^15.3e}")

        # Añadir fila a la tabla
        tabla_resultados.append([i, x1, f_x1, error])
        x_val.append(x1)

        # Verificar convergencia
        if error < tol:
            print("\nConvergencia alcanzada.")
            return x1, tabla_resultados, x_val

        # Comprobación de 'seguridad': si Newton sale de [a, b], abortamos.
        # (Newton no garantiza permanecer en el intervalo como sí hace la bisección.)
        if not a <= x1 <= b:
            print(f"x1 = {x1} está fuera del intervalo [{a}, {b}].")
            return None, tabla_resultados, x_val

        x0 = x1  # Preparar siguiente iteración

    # Si se consumen las iteraciones sin cumplir tolerancia
    print("\nNo se alcanzó la convergencia.")
    return None, tabla_resultados, x_val

# Función para guardar la tabla en un archivo CSV con formato claro
def guardar_csv(tabla, archivo_csv):
    """
    Guarda la tabla de iteraciones en un CSV con cabecera.

    Parámetros:
        tabla (list[list]): filas [i, x1, f(x1), error].
        archivo_csv (str): nombre de archivo de salida.
    """
    with open(archivo_csv, mode='w', newline='') as archivo:
        gen = csv.writer(archivo)
        gen.writerow(["Iteración", "x1", "f(x1)", "Error"])
        for fila in tabla:
            i, x1, fx1, error = fila
            gen.writerow([
                i,
                f"{x1:.10f}",
                f"{fx1:.10e}",
                f"{error:.10e}"
            ])

# Función que genera y guarda el gráfico de la función + iteraciones
def graficar_funcion_y_iteraciones(f_expr, x_vals, a, b, archivo_salida):
    """
    Dibuja f(x) en [a,b] y superpone las 'escaleras' del método de Newton.

    Parámetros:
        f_expr (sympy.Expr): expresión simbólica de f(x).
        x_vals (list[float]): iterados x0, x1, ..., xN (para marcar las escalas).
        a, b (float): rango para el eje x en el gráfico.
        archivo_salida (str): nombre del PNG a guardar.
    """
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr, 'numpy')

    # linspace para dibujar la curva de f(x) con suavidad
    x_plot = np.linspace(a, b, 500)
    y_plot = f(x_plot)

    plt.figure(figsize=(10, 6))
    # Eje x en y=0 para referencia visual
    plt.axhline(0, color='gray', linestyle='--')

    # Convertir la función a LaTeX para la leyenda
    formula_latex = r"$" + sp.latex(f_expr) + r"$"
    plt.plot(x_plot, y_plot, label=formula_latex, linewidth=2)

    # 'Escaleras' de Newton: vertical a la curva, horizontal al eje x, etc.
    for i in range(len(x_vals) - 1):
        xi = x_vals[i]
        xi_sig = x_vals[i + 1]
        # Segmento vertical: (xi, 0) -> (xi, f(xi))
        plt.plot([xi, xi], [0, f(xi)], color='red', linestyle='--')
        # Segmento horizontal: (xi, f(xi)) -> (xi+1, 0)
        plt.plot([xi, xi_sig], [f(xi), 0], color='red', linestyle='--')
        # Marca el punto en la curva
        plt.plot(xi, f(xi), 'ro')

    # Marca el último punto (aproximación final)
    plt.plot(x_vals[-1], f(x_vals[-1]), 'go', label='Aproximación final')

    plt.title("Método de Newton-Raphson")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig(archivo_salida)
    plt.close()

# Programa principal
if __name__ == "__main__":
    # Obtener fecha y hora actual en formato YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Método de Newton-Raphson\n")

    # Solicitar al usuario la función y parámetros
    # Usamos sympify en a, b, x0 para permitir entradas como 'pi'.
    funcion = input("Introduce la función f(x): ")

    a = float(sp.sympify(input("Extremo izquierdo del intervalo a (puedes usar pi): ")))
    b = float(sp.sympify(input("Extremo derecho del intervalo b (puedes usar pi): ")))
    x0 = float(sp.sympify(input("Punto inicial x0 (puedes usar pi): ")))

    # Resto de parámetros numéricos
    tol = float(input("Tolerancia (por ejemplo 1e-6): "))
    max_iter = int(input("Número máximo de iteraciones: "))

    # Verificar que el punto inicial está dentro del intervalo
    if not a <= x0 <= b:
        print("Error: x0 no está dentro del intervalo.")
    else:
        # Ejecutar método
        raiz, tabla, iteraciones = newton_metodo(funcion, x0, a, b, tol, max_iter)

        # Guardar CSV si hubo al menos una iteración
        if tabla:
            nom_csv = f"resultados_newton_{timestamp}.csv"
            guardar_csv(tabla, nom_csv)
            print("Tabla de resultados guardada en 'resultados_newton.csv'.")

        # Si se halló una raíz, generar gráfico
        if raiz is not None:
            nom_imagen = f"grafico_newton_{timestamp}.png"
            graficar_funcion_y_iteraciones(sp.sympify(funcion), iteraciones, a, b, nom_imagen)
            print("Gráfico guardado en 'grafico_newton.png'.")
            print(f"\nRaíz aproximada: {raiz}")
