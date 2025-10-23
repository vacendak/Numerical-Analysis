"""
Propósito:
  - Cargar un CSV ubicado en la misma carpeta del script.
  - Preguntar en qué columna están los valores aproximados p_n.
  - Calcular p_N como el último valor de esa columna (valor de referencia).
  - Construir una nueva tabla con:
      1) p_n                -> los valores originales de la columna elegida
      2) E_n = |p_n - p_N|  -> error absoluto respecto a p_N
      3) lnE_n = ln(E_n)    -> logaritmo natural del error (si E_n == 0 -> NaN)
      4) alpha_n            -> usando: alpha_n = [ln|p_{n+1}-p_N| - ln|p_n-p_N|] / [ln|p_n-p_N| - ln|p_{n-1}-p_N|]
         (definida solo cuando existen n-1, n, n+1 y los ln son finitos)
  - Graficar los puntos (ln(E_{n+1}), ln(E_n)) y ajustar una recta por mínimos cuadrados:
      y = m*x + b con y = ln(E_n), x = ln(E_{n+1}).
    Anotar en la figura la pendiente estimada m y el último alpha_n válido.
  - Guardar:
      - CSV con las columnas [p_n, E_n, lnE_n, alpha_n]
      - PNG con la gráfica

Uso:
  python orden_convergencia_alpha.py
  (Pide nombre del CSV, por ejemplo: datos.csv)
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ruta_en_misma_carpeta(nombre_archivo):
    """
    Devuelve la ruta absoluta en la misma carpeta que este script.
    Ignora subrutas introducidas por seguridad.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.basename(nombre_archivo)
    return os.path.join(script_dir, base)

def leer_csv_con_flexibilidad(path_csv):
    """
    Intenta leer un CSV con pandas.
    Si falla, intenta separador ';' y decimal ','.
    """
    try:
        return pd.read_csv(path_csv)
    except Exception:
        pass
    try:
        return pd.read_csv(path_csv, sep=";", decimal=",")
    except Exception as e:
        raise RuntimeError("No se pudo leer el CSV: " + str(e))

def elegir_columna(df):
    """
    Pide al usuario la columna de aproximaciones p_n.
    Acepta coincidencia ignorando mayúsculas.
    """
    cols = list(df.columns)
    print("Columnas disponibles:")
    for i, c in enumerate(cols, 1):
        print("{:2d}. {}".format(i, c))
    while True:
        nombre = input("Nombre de la columna con las aproximaciones p_n: ").strip()
        if nombre in cols:
            return nombre
        m = [c for c in cols if c.lower() == nombre.lower()]
        if len(m) == 1:
            return m[0]
        print("No coincide con ninguna columna. Intenta de nuevo.")

def serie_a_numerico(serie):
    """
    Convierte serie a numérico de forma robusta:
      - Reemplaza coma decimal por punto si es texto.
      - Convierte a float con coerción (no convertibles -> NaN).
    """
    s = serie.copy()
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def construir_tabla_pn_En_lnEn_alpha(pvals):
    """
    Recibe un array o serie de p_n (float).
    Calcula:
      p_N        -> último valor
      E_n        -> |p_n - p_N|
      lnE_n      -> ln(E_n) si E_n > 0, en otro caso NaN
      alpha_n    -> definida para n = 1..N-2 si lnE_{n-1}, lnE_n, lnE_{n+1} son finitos

    Devuelve:
      df_out, ultimo_alpha_valido (float o NaN)
    """
    pvals = np.asarray(pvals, dtype=float)
    N = len(pvals)
    if N < 3:
        raise ValueError("Se requieren al menos 3 valores p_n para calcular alpha_n.")

    pN = pvals[-1]
    En = np.abs(pvals - pN)

    # lnE_n: definir solo si E_n > 0
    lnEn = np.full(N, np.nan, dtype=float)
    mask_pos = En > 0
    lnEn[mask_pos] = np.log(En[mask_pos])
    # Nota: si E_n == 0, ln no está definido; se deja como NaN.

    # alpha_n para n = 1..N-2
    alpha = np.full(N, np.nan, dtype=float)
    for n in range(1, N - 1):
        # necesitamos n-1, n, n+1
        a = lnEn[n + 1]
        b = lnEn[n]
        c = lnEn[n - 1]
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
            denom = (b - c)
            if denom != 0.0:
                alpha[n] = (a - b) / denom
            else:
                alpha[n] = np.nan

    # último alpha válido
    idx_validos = np.where(np.isfinite(alpha))[0]
    ultimo_alpha = alpha[idx_validos[-1]] if idx_validos.size > 0 else np.nan

    df_out = pd.DataFrame({
        "p_n": pvals,
        "E_n": En,
        "lnE_n": lnEn,
        "alpha_n": alpha
    })
    return df_out, ultimo_alpha

def pares_lnEn_para_ajuste(df_out):
    """
    Construye los pares (x, y) con:
      x = ln(E_{n+1})
      y = ln(E_n)
    usando la columna lnE_n del df_out.
    Excluye filas donde falten valores.
    """
    lnE = df_out["lnE_n"].to_numpy()
    x = lnE[1:]
    y = lnE[:-1]
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

def ajuste_lineal(x, y):
    """
    Ajuste por mínimos cuadrados y -> y = m*x + b.
    Devuelve m, b, R^2 y y_pred.
    """
    if len(x) < 2:
        raise ValueError("Insuficientes puntos para el ajuste lineal (>= 2).")
    coef = np.polyfit(x, y, 1)
    m, b = coef[0], coef[1]
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return m, b, r2, y_pred

def grafica_lnEn_pairs(x, y, y_pred, nombre_csv, nombre_columna, ultimo_alpha, salida_png):
    """
    Grafica los puntos (ln(E_{n+1}), ln(E_n)) y la recta de tendencia.
    Anota la pendiente m y el último alpha_n válido.
    """
    plt.figure(figsize=(7.5, 6.0))
    plt.scatter(x, y, s=28, alpha=0.9, edgecolor="black", linewidth=0.5)

    # Dibujar la recta en orden creciente de x
    orden = np.argsort(x)
    plt.plot(x[orden], y_pred[orden], linewidth=2.0)

    # Evitar .format por las llaves en n+1
    plt.title(f"Diagrama ln(E_{{n+1}}) vs ln(E_n)")
    plt.xlabel("ln(E_{n+1})")
    plt.ylabel("ln(E_n)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    # Márgenes amigables
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    dx = max(1e-6, 0.05 * (x_max - x_min if x_max > x_min else 1.0))
    dy = max(1e-6, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
    plt.xlim(x_min - dx, x_max + dx)
    plt.ylim(y_min - dy, y_max + dy)

    # Anotación con pendiente y último alpha
    # Usamos sintaxis de TeX para alpha
    ax = plt.gca()
    texto = r"$\text{pendiente } m = " + f"{(np.polyfit(x, y, 1)[0]):.6f}" + r"$"  # m mostrado
    if np.isfinite(ultimo_alpha):
        texto += "\n" + r"$\alpha_{\mathrm{ultimo}} = " + f"{ultimo_alpha:.6f}" + r"$"
    ax.text(
        0.03, 0.97, texto,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black")
    )

    plt.tight_layout()
    plt.savefig(salida_png, dpi=160)
    plt.close()

def main():
    # Pedir nombre del CSV (en la misma carpeta del script)
    if len(sys.argv) >= 2:
        nombre_csv = sys.argv[1]
    else:
        nombre_csv = input("Nombre del CSV (ejemplo: datos.csv): ").strip()

    ruta = ruta_en_misma_carpeta(nombre_csv)
    if not os.path.isfile(ruta):
        print("No existe el archivo en la carpeta del script: {}".format(ruta))
        sys.exit(1)

    df = leer_csv_con_flexibilidad(ruta)

    # Elegir columna con p_n
    col = elegir_columna(df)

    # Convertir a numérico y filtrar NaN
    p_series = serie_a_numerico(df[col])
    total_ini = len(p_series)
    p_series = p_series.dropna()
    n_nans = total_ini - len(p_series)

    if len(p_series) < 3:
        print("Muy pocos datos en la columna seleccionada tras filtrar NaN. Necesarios >= 3.")
        sys.exit(1)

    # Construir tabla con p_n, E_n, lnE_n, alpha_n
    tabla, ultimo_alpha = construir_tabla_pn_En_lnEn_alpha(p_series.values)

    # Pares para ajuste: (ln(E_{n+1}), ln(E_n))
    x, y = pares_lnEn_para_ajuste(tabla)
    if len(x) < 2:
        print("Insuficientes pares válidos (ln(E_{n+1}), ln(E_n)) para el ajuste (>= 2).")
        # Guardamos la tabla igualmente
        pass

    # Ajuste lineal si hay datos suficientes
    if len(x) >= 2:
        m, b, r2, y_pred = ajuste_lineal(x, y)
    else:
        m, b, r2, y_pred = np.nan, np.nan, np.nan, np.array([])

    # Salidas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_csv = os.path.splitext(os.path.basename(ruta))[0]
    fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    etiqueta_col = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in col)

    salida_tabla = os.path.join(script_dir, f"{base_csv}_{etiqueta_col}_tabla_error_alpha_{fecha}.csv")
    salida_png   = os.path.join(script_dir, f"{base_csv}_{etiqueta_col}_lnEn_pairs_{fecha}.png")

    # Guardar tabla
    tabla.to_csv(salida_tabla, index=False, float_format="%.8f")

    # Graficar si hay datos suficientes
    if len(x) >= 2:
        grafica_lnEn_pairs(x, y, y_pred, base_csv, col, ultimo_alpha, salida_png)

    # Informe por consola
    print("Columna usada (p_n): {}".format(col))
    print("Filtrados NaN en p_n: {}".format(n_nans))
    print("Valor p_N (último de la columna): {:.12g}".format(p_series.values[-1]))
    print("Filas en tabla generada: {}".format(len(tabla)))
    print("Último alpha_n válido:", "NaN" if not np.isfinite(ultimo_alpha) else f"{ultimo_alpha:.6f}")

    if len(x) >= 2:
        print("Ajuste lineal sobre pares (ln(E_{n+1}), ln(E_n)):")
        print("  Pendiente m:  {:.6f}".format(m))
        print("  Ordenada b:   {:.6f}".format(b))
        print("  R^2:          {:.6f}".format(r2))
        print("Figura guardada en:", salida_png)
    else:
        print("No se pudo generar la figura por falta de pares válidos (ln(E_{n+1}), ln(E_n)).")

    print("Tabla guardada en:", salida_tabla)
    print("\nNotas:")
    print("- Si alguna fila tiene E_n = 0, su ln(E_n) es indefinido y se marca como NaN.")
    print("- alpha_n solo se define cuando existen n-1, n y n+1 y los ln(E) son finitos.")
    print("- La pendiente m del ajuste en el diagrama ln(E_{n+1}) vs ln(E_n) es otra forma de estimar el comportamiento asintótico.")

if __name__ == "__main__":
    main()
