# ==============================================================
# Factorización LU (Método de Doolittle)
# --------------------------------------------------------------
# L unitaria (diag(L)=1). Se factoriza A = L U y se resuelve LU x = b.
# Se cuentan operaciones:
#   - mult/div = multiplicaciones + divisiones
#   - sum/rest = sumas + restas
# Entrada: CSV sin cabecera, separador coma, con n filas y n+1 columnas (A|b)
# Salida: imprime L, U, y, x, error; guarda L.csv, U.csv, y.csv, x.csv (.5g)
# ==============================================================

import pandas as pd

# -------------------- Entrada CSV -----------------------------
def leer_csv(nombre):
    df = pd.read_csv(nombre, header=None)
    n, m = df.shape
    if m != n + 1:
        raise ValueError("El CSV debe tener n filas y n+1 columnas (A|b).")
    A = df.iloc[:, :-1].to_numpy().tolist()
    b = df.iloc[:, -1].to_numpy().tolist()
    return A, b

# -------------------- Crear matriz ----------------------------
def crear_matriz(n, v=0.0):
    return [[v for _ in range(n)] for _ in range(n)]

# -------------------- Factorización (Doolittle) --
def factorizacion_LU_doolittle(A, eps=1e-14):
    """
    Doolittle: L unitaria, U superior.
    Devuelve L, U y el conteo (mult/div, sum/rest).
    Lanza ValueError si aparece pivote U[i][i] ~ 0.
    """
    n = len(A)
    L = crear_matriz(n)
    U = crear_matriz(n)

    multdiv = 0  # multiplicaciones + divisiones
    sumrest = 0  # sumas + restas

    # L con diagonal unitaria
    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        # ---- Fila i de U: j = i..n-1
        for j in range(i, n):
            # U[i][j] = A[i][j] - sum_{k=0}^{i-1} L[i][k]*U[k][j]
            temp = 0.0
            for k in range(i):
                temp += L[i][k] * U[k][j]
                multdiv += 1          # multiplicación
                sumrest += 1          # suma acumulada
            if i > 0:
                sumrest += 1          # resta final A - temp
            U[i][j] = A[i][j] - temp

        # Comprobar pivote antes de dividir
        if abs(U[i][i]) < eps:
            raise ValueError(f"Pivote nulo o casi nulo en U[{i},{i}]. "
                             "La factorización sin pivoteo no es posible.")

        # ---- Columna i de L: j = i+1..n-1
        for j in range(i + 1, n):
            # L[j][i] = (A[j][i] - sum_{k=0}^{i-1} L[j][k]*U[k][i]) / U[i][i]
            temp = 0.0
            for k in range(i):
                temp += L[j][k] * U[k][i]
                multdiv += 1          # multiplicación
                sumrest += 1          # suma acumulada
            if i > 0:
                sumrest += 1          # resta previa
            L[j][i] = (A[j][i] - temp) / U[i][i]
            multdiv += 1              # división

    return L, U, multdiv, sumrest

# -------------------- Sustituciones ---------------------------
def sustitucion_progresiva(L, b):
    """
    L unitaria: y[i] = b[i] - sum_{k=0}^{i-1} L[i][k]*y[k]
    No hay división.
    """
    n = len(L)
    y = [0.0] * n
    multdiv = 0
    sumrest = 0
    for i in range(n):
        temp = 0.0
        for k in range(i):
            temp += L[i][k] * y[k]
            multdiv += 1
            sumrest += 1
        if i > 0:
            sumrest += 1
        y[i] = b[i] - temp
    return y, multdiv, sumrest

def sustitucion_regresiva(U, y):
    """
    U triangular superior: x[i] = (y[i] - sum_{k=i+1}^{n-1} U[i][k]*x[k]) / U[i][i]
    Hay una división por fila.
    """
    n = len(U)
    x = [0.0] * n
    multdiv = 0
    sumrest = 0
    for i in range(n - 1, -1, -1):
        temp = 0.0
        for k in range(i + 1, n):
            temp += U[i][k] * x[k]
            multdiv += 1
            sumrest += 1
        if i < n - 1:
            sumrest += 1
        x[i] = (y[i] - temp) / U[i][i]
        multdiv += 1
    return x, multdiv, sumrest

# -------------------- Verificación ----------------------------
def producto(A, B):
    n = len(A)
    C = crear_matriz(n)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def error_factorizacion(A, L, U):
    R = producto(L, U)
    n = len(A)
    err = 0.0
    for i in range(n):
        for j in range(n):
            d = abs(A[i][j] - R[i][j])
            if d > err:
                err = d
    return err

# -------------------- Programa principal ----------------------
def main():
    archivo = input("Archivo CSV (A|b, coma, sin cabecera): ").strip()
    A, b = leer_csv(archivo)
    n = len(A)

    # Factorización Doolittle
    L, U, m1, s1 = factorizacion_LU_doolittle(A)

    # Sustituciones
    y, m2, s2 = sustitucion_progresiva(L, b)
    x, m3, s3 = sustitucion_regresiva(U, y)

    # Conteos por fase
    multdiv_fact = m1
    sumrest_fact = s1
    multdiv_sust = m2 + m3
    sumrest_sust = s2 + s3

    # Totales
    multdiv_total = multdiv_fact + multdiv_sust
    sumrest_total = sumrest_fact + sumrest_sust
    total_ops = multdiv_total + sumrest_total

    # Costes teóricos (aprox.)
    teor_fact = (1/3) * n**3
    teor_sust = 2 * n**2
    teor_total = teor_fact + teor_sust

    # Salida
    print("\nL =")
    for fila in L:
        print("  ", *[f"{v:12.6g}" for v in fila])
    print("\nU =")
    for fila in U:
        print("  ", *[f"{v:12.6g}" for v in fila])
    print("\ny =", " ".join(f"{v:12.6g}" for v in y))
    print("x =", " ".join(f"{v:12.6g}" for v in x))

    err = error_factorizacion(A, L, U)
    print(f"\nError máximo |A - L*U| = {err:.3e}")

    # Resumen de costes
    print("\n=== COSTE COMPUTACIONAL (mult/div y sum/rest) ===")

    print(f"mult/div: {multdiv_sust}   \nsum/rest: {sumrest_sust}   \ntotal: {multdiv_sust + sumrest_sust}")
    print(f"Teórico (2 * n^2): {teor_sust:.0f}   \nRelación: {(multdiv_sust + sumrest_sust)/teor_sust:.2f}")

    # Guardado CSV (.5g)
    pd.DataFrame([[f"{v:.5g}" for v in fila] for fila in L]).to_csv("L.csv", index=False, header=False)
    pd.DataFrame([[f"{v:.5g}" for v in fila] for fila in U]).to_csv("U.csv", index=False, header=False)
    pd.DataFrame([[f"{v:.5g}"] for v in y]).to_csv("y.csv", index=False, header=False)
    pd.DataFrame([[f"{v:.5g}"] for v in x]).to_csv("x.csv", index=False, header=False)
    print("\nArchivos guardados con formato .5g: L.csv, U.csv, y.csv, x.csv")

if __name__ == "__main__":
    main()
