# ==============================================================
# Factorización LU con l_ii = u_ii, manteniendo A = L U.
# --------------------------------------------------------------
# - Conteo de operaciones (mult/div y sum/rest) SOLO del
#   algoritmo de factorización y sustituciones
# - Lectura A|b desde CSV (coma, sin cabecera).
# - Comparación con costes teóricos: fact ~(1/3) n^3, sust ~ 2 n^2.
# - Guarda L.csv, U.csv, y.csv, x.csv en formato .5g.
# ==============================================================

import math
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

# -------------------- Utilidades ------------------------------
def crear_matriz(n, v=0.0):
    return [[v for _ in range(n)] for _ in range(n)]

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

# -------------------- Factorización LU -----------------
def factorizacion_LU(A, eps=1e-14):
    n = len(A)
    L = crear_matriz(n)
    U = crear_matriz(n)

    multdiv = 0  # multiplicaciones + divisiones
    sumrest = 0  # sumas + restas

    # Diagonal unitaria de L
    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        # Fila i de U
        for j in range(i, n):
            tmp = 0.0
            for k in range(i):
                tmp += L[i][k] * U[k][j]
                multdiv += 1
                sumrest += 1
            if i > 0:
                sumrest += 1  # resta A - tmp
            U[i][j] = A[i][j] - tmp

        # Comprobación de pivote
        if abs(U[i][i]) < eps:
            raise ValueError(f"Pivote nulo o casi nulo en U[{i},{i}]. "
                             "La factorización no es posible.")

        # Columna i de L
        for j in range(i + 1, n):
            tmp = 0.0
            for k in range(i):
                tmp += L[j][k] * U[k][i]
                multdiv += 1
                sumrest += 1
            if i > 0:
                sumrest += 1
            L[j][i] = (A[j][i] - tmp) / U[i][i]
            multdiv += 1  # división

    return L, U, multdiv, sumrest

# ------------- l_ii = u_ii -------------
def igualar_diagonales(L, U, eps=1e-14):
    """
    A=L U -> A = L2 U2 con diag(L2) = diag(U2).
    Procedimiento:
      1) S1 = diag(sign(U_ii))  -> U1 = S1 U  (no negativas)
         L1 = L S1
      2) S2 = diag(sqrt(|U1_ii|)) -> L2 = L1 S2 ; U2 = S2^{-1} U1
    Requiere U_ii != 0. Devuelve L2, U2.
    """
    n = len(L)

    # Construir S1 (signos) y S2 (raíz de la diagonal no negativa)
    s1 = [1.0] * n
    s2 = [0.0] * n
    for i in range(n):
        ui = U[i][i]
        if abs(ui) < eps:
            raise ValueError(f"No se puede imponer l_ii=u_ii: u[{i},{i}] ~ 0.")
        s1[i] = 1.0 if ui >= 0.0 else -1.0
        s2[i] = math.sqrt(abs(ui))

    # L1 = L * S1
    L1 = [fila[:] for fila in L]
    for i in range(n):
        si = s1[i]
        for r in range(n):
            L1[r][i] *= si

    # U1 = S1 * U
    U1 = [fila[:] for fila in U]
    for i in range(n):
        si = s1[i]
        for c in range(n):
            U1[i][c] *= si

    # L2 = L1 * S2
    L2 = [fila[:] for fila in L1]
    for i in range(n):
        si = s2[i]
        for r in range(n):
            L2[r][i] *= si

    # U2 = S2^{-1} * U1
    U2 = [fila[:] for fila in U1]
    for i in range(n):
        inv = 1.0 / s2[i]
        for c in range(n):
            U2[i][c] *= inv

    # Ahora diag(L2) = diag(U2) = sqrt(|diag(U)|)
    return L2, U2

# -------------------- Sustituciones ---------------------------
def sustitucion_progresiva(L, b):
    """
    Resuelve L*y = b para L triangular inferior
    """
    n = len(L)
    y = [0.0] * n
    multdiv = 0
    sumrest = 0
    for i in range(n):
        tmp = 0.0
        for k in range(i):
            tmp += L[i][k] * y[k]
            multdiv += 1
            sumrest += 1
        sumrest += 1
        y[i] = (b[i] - tmp) / L[i][i]
        multdiv += 1  # división por L[i][i]
    return y, multdiv, sumrest

def sustitucion_regresiva(U, y):
    """
    Resuelve U*x = y para U triangular superior
    """
    n = len(U)
    x = [0.0] * n
    multdiv = 0
    sumrest = 0
    for i in range(n - 1, -1, -1):
        tmp = 0.0
        for k in range(i + 1, n):
            tmp += U[i][k] * x[k]
            multdiv += 1
            sumrest += 1
        if i < n - 1:
            sumrest += 1
        else:
            sumrest += 1  # también contamos y[i]-tmp cuando tmp=0
        x[i] = (y[i] - tmp) / U[i][i]
        multdiv += 1
    return x, multdiv, sumrest

# -------------------- Programa principal ----------------------
def main():
    archivo = input("Archivo CSV (A|b, coma, sin cabecera): ").strip()
    A, b = leer_csv(archivo)
    n = len(A)

    # 1) Factorización Doolittle SIN pivoteo (conteo del algoritmo)
    L_raw, U_raw, m1, s1 = factorizacion_LU(A)

    # 2) Reescalar para imponer l_ii = u_ii (NO se cuenta en el coste)
    L, U = igualar_diagonales(L_raw, U_raw)

    # 3) Sustituciones con L y U reescaladas (conteo)
    y, m2, s2 = sustitucion_progresiva(L, b)
    x, m3, s3 = sustitucion_regresiva(U, y)

    # --- Conteos por fase ---
    multdiv_fact = m1
    sumrest_fact = s1
    multdiv_sust = m2 + m3
    sumrest_sust = s2 + s3

    multdiv_total = multdiv_fact + multdiv_sust
    sumrest_total = sumrest_fact + sumrest_sust
    total_ops = multdiv_total + sumrest_total

    # --- Costes teóricos (aprox.) ---
    teor_fact = (1/3) * n**3
    teor_sust = 2 * n**2
    teor_total = teor_fact + teor_sust

    # --- Salida ---
    print("\nL, con diag(L)=diag(U)) =")
    for fila in L:
        print("  ", *[f"{v:12.6g}" for v in fila])
    print("\nU =")
    for fila in U:
        print("  ", *[f"{v:12.6g}" for v in fila])
    print("\ny =", " ".join(f"{v:12.6g}" for v in y))
    print("x =", " ".join(f"{v:12.6g}" for v in x))

    err = error_factorizacion(A, L, U)
    print(f"\nError máximo |A - L*U| = {err:.3e}")

    print("\n=== COSTE COMPUTACIONAL (mult/div y sum/rest) ===")
    # print("--- FACTORIZACIÓN ---")
    # print(f"mult/div: {multdiv_fact}   sum/rest: {sumrest_fact}   total: {multdiv_fact + sumrest_fact}")
    # print(f"Teórico (1/3 n^3): {teor_fact:.0f}   Relación: {(multdiv_fact + sumrest_fact)/teor_fact:.2f}")

    print("\n--- SUSTITUCIONES (Ly=b y Ux=y, con L reescalada) ---")
    print(f"mult/div: {multdiv_sust}   \nsum/rest: {sumrest_sust}   \ntotal: {multdiv_sust + sumrest_sust}")
    print(f"Teórico (2 n^2): {teor_sust:.0f}   \nRelación: {(multdiv_sust + sumrest_sust)/teor_sust:.2f}")

    # print("\n--- TOTAL ---")
    # print(f"Total real: {total_ops}   Teórico total: {teor_total:.0f}   Relación: {total_ops/teor_total:.2f}")

    # Guardado CSV (.5g)
    pd.DataFrame([[f"{v:.5g}" for v in fila] for fila in L]).to_csv("L.csv", index=False, header=False)
    pd.DataFrame([[f"{v:.5g}" for v in fila] for fila in U]).to_csv("U.csv", index=False, header=False)
    pd.DataFrame([[f"{v:.5g}"] for v in y]).to_csv("y.csv", index=False, header=False)
    pd.DataFrame([[f"{v:.5g}"] for v in x]).to_csv("x.csv", index=False, header=False)
    print("\nArchivos guardados con formato .5g: L.csv, U.csv, y.csv, x.csv")

if __name__ == "__main__":
    main()
