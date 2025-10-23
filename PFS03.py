"""
# -------------------------------------------------------------
# Método del Punto Fijo con opción Steffensen
# -------------------------------------------------------------
Descripcion:
  Metodo de Punto Fijo para aproximar ceros de un polinomio p(x) con g(x) definida por el usuario.
  - Entradas directas por consola: p(x), g(x), x0, tolerancia, max. iteraciones.
  - Muestra tabla de iteraciones en pantalla.
  - Guarda CSV con la tabla.
  - Dibuja un grafico sencillo de la evolucion de x_n y |p(x_n)| y lo guarda como PNG.
  - Opcion opcional: ejecutar tambien la version acelerada de Steffensen y guardar salidas separadas.

Requisitos: sympy, matplotlib
Nota: Este archivo usa solo caracteres ASCII para evitar problemas con listings en LaTeX.
"""

import csv
from datetime import datetime
import sympy as sp
import matplotlib.pyplot as plt

# ==========================
#   FUNCIONES AUXILIARES
# ==========================

def normalizar_expresion(texto: str) -> str:
    """Convierte '^' en '**' (para exponentes)."""
    return texto.replace('^', '**')


def construir_polinomio(polinomio_str: str):
    """Devuelve x (simbolo), p_expr (sympy) y p_num (funcion numerica)."""
    x = sp.symbols('x')
    p_expr = sp.sympify(normalizar_expresion(polinomio_str))
    p_num = sp.lambdify(x, p_expr, 'math')
    return x, p_expr, p_num


def construir_g(g_str: str):
    """Devuelve g_expr (sympy) y g_num (funcion numerica) desde el texto de g(x)."""
    x = sp.symbols('x')
    g_expr = sp.sympify(normalizar_expresion(g_str))
    g_num = sp.lambdify(x, g_expr, 'math')
    return g_expr, g_num

# ==========================
#       PUNTO FIJO
# ==========================

def punto_fijo(g, p, x0, tolerancia, max_iter):
    """Devuelve (raiz_aprox, historial) con historial como lista de tuplas:
       (iteracion, xn, gxn, residuo_abs, error_abs)"""
    historial = []
    xn = float(x0)
    gxn = float(g(xn))
    residuo = abs(float(p(xn)))
    historial.append((0, xn, gxn, residuo, float('nan')))

    for k in range(1, max_iter + 1):
        xsig = float(g(xn))
        err = abs(xsig - xn)
        residuo = abs(float(p(xsig)))
        gxsig = float(g(xsig))
        historial.append((k, xsig, gxsig, residuo, err))
        if err < tolerancia or residuo < tolerancia:
            return xsig, historial
        xn = xsig

    return xn, historial


def punto_fijo_steffensen(g, p, x0, tolerancia, max_iter):
    """Aceleracion de Aitken/Steffensen sobre la iteracion de punto fijo.
    Formula (usando Delta):
        x_{n+1} = x_n - (Delta x_n)^2 / (Delta^2 x_n)
    con
        Delta x_n   = g(x_n) - x_n
        Delta^2 x_n = g(g(x_n)) - 2*g(x_n) + x_n
    Si el denominador es muy pequeno, se usa el paso normal x_{n+1} = g(x_n).
    Devuelve (raiz_aprox, historial) con el mismo formato que punto_fijo.
    """
    eps_den = 1e-15
    historial = []
    xn = float(x0)

    gxn = float(g(xn))
    residuo = abs(float(p(xn)))
    historial.append((0, xn, gxn, residuo, float('nan')))

    for k in range(1, max_iter + 1):
        gx = float(g(xn))
        ggx = float(g(gx))
        delta = gx - xn
        delta2 = ggx - 2*gx + xn
        if abs(delta2) > eps_den:
            xsig = xn - (delta*delta)/delta2
        else:
            xsig = gx
        err = abs(xsig - xn)
        residuo = abs(float(p(xsig)))
        gxsig = float(g(xsig))
        historial.append((k, xsig, gxsig, residuo, err))
        if err < tolerancia or residuo < tolerancia:
            return xsig, historial
        xn = xsig

    return xn, historial

# ==========================
#          SALIDA
# ==========================

def guardar_tabla_csv(historial, base):
    fecha = datetime.now().strftime('%Y%m%d_%H%M%S')
    archivo = f"tabla_iteraciones_{base}_{fecha}.csv"
    with open(archivo, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["iteración", "xn", "g(xn)", "|p(xn)|", "|Delta x|"])
        for it, xn, gxn, res, err in historial:
            w.writerow([it, f"{xn:.8g}", f"{gxn:.8g}", f"{res:.8g}", f"{err:.8g}"])
    return archivo


def graficar_evolucion(historial, base, titulo):
    its = [r[0] for r in historial]
    xns = [r[1] for r in historial]
    residuos = [r[3] for r in historial]

    fig = plt.figure()
    plt.plot(its, xns, marker='o', label='x_n')
    plt.plot(its, residuos, marker='s', linestyle='--', label='|p(x_n)|')
    plt.xlabel('iteración')
    plt.ylabel('Valor')
    plt.title(titulo)
    plt.grid(True, which='both', linestyle=':')
    plt.legend()

    nombre_png = f"evolución_punto_fijo_{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(nombre_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return nombre_png

# =====================================
#            PROGRAMA PRINCIPAL
# =====================================

def main():
    print("===== MÉTODO DE PUNTO FIJO (simple) =====")
    polinomio_str = input("Introduce el polinomio p(x) (ej.: x^3 - 2*x - 5): ").strip()
    if not polinomio_str:
        raise ValueError("Debe introducir un polinomio.")

    g_str = input("Introduce g(x) (ej.: (x+2)/3): ").strip()
    if not g_str:
        raise ValueError("Debe introducir g(x).")

    # Entradas numericas sencillas (admite decimales y notacion cientifica en tolerancia)
    x0 = float(input("Introduce x0 (semilla inicial): "))
    tolerancia = float(input("Introduce la tolerancia (ej.: 1e-8): "))
    max_iter = int(input("Introduce el número maximo de iteraciones: "))

    usar_steff = input("Ejecutar también versión acelerada (Steffensen)? (s/n): ").strip().lower() == 's'

    # Construccion simbolica y numerica
    x, p_expr, p_num = construir_polinomio(polinomio_str)
    g_expr, g_num = construir_g(g_str)

    # Ejecuta el metodo (basico)
    raiz_aprox, historial = punto_fijo(g_num, p_num, x0, tolerancia, max_iter)

    # (Opcional) ejecutar version acelerada
    raiz_aprox_acc = None
    historial_acc = None
    if usar_steff:
        raiz_aprox_acc, historial_acc = punto_fijo_steffensen(g_num, p_num, x0, tolerancia, max_iter)

    # Salidas en consola
    print("Parámetros:")
    print(f"  p(x) = {sp.simplify(p_expr)}")
    print(f"  g(x) = {sp.simplify(g_expr)}")
    print(f"  x0 = {x0}")
    print(f"  tolerancia = {tolerancia}")
    print(f"  max_iter = {max_iter}")

    encabezado = f"{'it':>3} | {'xn':>20} | {'g(xn)':>20} | {'|p(xn)|':>12} | {'|Delta x|':>12}"
    print("Tabla de iteraciones (básico):")
    print(encabezado)
    print('-' * len(encabezado))
    for it, xn, gxn, res, err in historial:
        print(f"{it:3d} | {xn:>20.12g} | {gxn:>20.12g} | {res:>12.4e} | {err:>12.4e}")

    print("Resultado (básico):")
    print(f"  Aproximación a la raiz = {raiz_aprox:.16g}")
    print(f"  Residuo |p(raiz)|     = {abs(float(p_num(raiz_aprox))):.4e}")
    print(f"  Iteraciones realizadas = {len(historial)-1}")

    # Guardado con nombres separados para comparar
    base = 'puntofijo'
    archivo_csv = guardar_tabla_csv(historial, base + '_basico')
    titulo = f"Evolución punto fijo (básico)"
    archivo_png = graficar_evolucion(historial, base + '_básico', titulo)

    if usar_steff:
        print("Tabla de iteraciones (Steffensen):")
        print(encabezado)
        print('-' * len(encabezado))
        for it, xn, gxn, res, err in historial_acc:
            print(f"{it:3d} | {xn:>20.12g} | {gxn:>20.12g} | {res:>12.4e} | {err:>12.4e}")

        print("Resultado (Steffensen):")
        print(f"  Aproximación a la raiz = {raiz_aprox_acc:.16g}")
        print(f"  Residuo |p(raiz)|     = {abs(float(p_num(raiz_aprox_acc))):.4e}")
        print(f"  Iteraciones realizadas = {len(historial_acc)-1}")

        archivo_csv_acc = guardar_tabla_csv(historial_acc, base + '_steffensen')
        titulo_acc = f"Evolución punto fijo (Steffensen)"
        archivo_png_acc = graficar_evolucion(historial_acc, base + '_steffensen', titulo_acc)

    print("Archivos generados:")
    print(f"  CSV  -> {archivo_csv}")
    print(f"  PNG  -> {archivo_png}")
    if usar_steff:
        print(f"  CSV  -> {archivo_csv_acc}")
        print(f"  PNG  -> {archivo_png_acc}")
    print("Listo.")

if __name__ == '__main__':
    main()
