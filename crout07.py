# ==============================================================
# FACTORIZACIÓN LU (Método de Crout)
# --------------------------------------------------------------
# Este programa resuelve el sistema Ax = b mediante factorización
# LU según el método de Crout, donde la matriz U tiene la
# diagonal unitaria (U con 1s en la diagonal).
#
# Entrada: archivo CSV (sin cabecera) con n filas y n+1 columnas
# (las n primeras corresponden a A y la última a b)
#
# Salida:
#   - Matrices L y U en pantalla
#   - Vectores y y x
#   - Error máximo de factorización |A - L*U|
#   - Archivos CSV: L.csv, U.csv, x.csv y y.csv (formato .5g)
# ==============================================================
import pandas as pd

# --------------------------------------------------------------
# Lectura de datos desde archivo CSV
# Formato esperado: n filas y n+1 columnas (A | b) sin cabecera.
# --------------------------------------------------------------
def leer_csv(nombre):
    df = pd.read_csv(nombre, header=None)
    n, m = df.shape
    if m != n + 1:
        raise ValueError("El CSV debe tener n filas y n+1 columnas (A|b).")
    A = df.iloc[:, :-1].to_numpy().tolist()
    b = df.iloc[:, -1].to_numpy().tolist()
    return A, b

# --------------------------------------------------------------
# Crea una matriz n x n inicializada con un valor v
# --------------------------------------------------------------
def crear_matriz(n, v=0.0):
    return [[v for _ in range(n)] for _ in range(n)]

# --------------------------------------------------------------
# FACTORIZACIÓN LU (Crout) con conteo de operaciones aritméticas
# U tiene diagonal unitaria, L contiene los pivotes
# --------------------------------------------------------------
def factorizacion_LU_crout(A):
    n = len(A)
    L = crear_matriz(n)
    U = crear_matriz(n)

    mults = 0  # multiplicaciones
    divs  = 0  # divisiones
    sums  = 0  # sumas/restas

    # Paso 1: inicialización
    U[0][0] = 1.0
    L[0][0] = A[0][0]

    # Paso 2: primera fila de U y primera columna de L
    for j in range(1, n):
        U[0][j] = A[0][j] / L[0][0]
        L[j][0] = A[j][0]
        divs += 1  # división

    # Pasos 3 a 5: cálculo de elementos internos de L y U
    for i in range(1, n - 1):
        # Cálculo del pivote L[i][i]
        temp = 0.0
        for k in range(i):
            temp += L[i][k] * U[k][i]
            mults += 1
            sums  += 1
        L[i][i] = A[i][i] - temp
        sums += 1  # resta final
        U[i][i] = 1.0

        # Cálculo de los elementos siguientes
        for j in range(i + 1, n):
            # U[i][j]
            temp1 = 0.0
            for k in range(i):
                temp1 += L[i][k] * U[k][j]
                mults += 1
                sums  += 1
            U[i][j] = (A[i][j] - temp1) / L[i][i]
            sums += 1   # resta previa
            divs += 1   # división

            # L[j][i]
            temp2 = 0.0
            for k in range(i):
                temp2 += L[j][k] * U[k][i]
                mults += 1
                sums  += 1
            L[j][i] = A[j][i] - temp2
            sums += 1  # resta final

    # Paso 6: última diagonal de L
    i = n - 1
    temp3 = 0.0
    for k in range(n - 1):
        temp3 += L[i][k] * U[k][i]
        mults += 1
        sums  += 1
    L[i][i] = A[i][i] - temp3
    sums += 1
    U[i][i] = 1.0

    return L, U, mults, divs, sums

# --------------------------------------------------------------
# Sustitución progresiva: resuelve L*y = b
# --------------------------------------------------------------
def sustitucion_progresiva(L, b):
    n = len(L)
    y = [0.0] * n
    mults = divs = sums = 0
    for i in range(n):
        temp = 0.0
        for k in range(i):
            temp += L[i][k] * y[k]
            mults += 1
            sums  += 1
        y[i] = (b[i] - temp) / L[i][i]
        sums += 1   # resta
        divs += 1   # división
    return y, mults, divs, sums

# --------------------------------------------------------------
# Sustitución regresiva: resuelve U*x = y
# --------------------------------------------------------------
def sustitucion_regresiva(U, y):
    n = len(U)
    x = [0.0] * n
    mults = divs = sums = 0
    for i in range(n - 1, -1, -1):
        temp = 0.0
        for k in range(i + 1, n):
            temp += U[i][k] * x[k]
            mults += 1
            sums  += 1
        x[i] = (y[i] - temp) / U[i][i]
        sums += 1   # resta
        divs += 1   # división (aunque U[i][i]=1, se cuenta igualmente)
    return x, mults, divs, sums

# --------------------------------------------------------------
# Producto de matrices cuadradas (para verificar el error)
# --------------------------------------------------------------
def producto(A, B):
    n = len(A)
    C = crear_matriz(n)
    for i in range(n):
        for j in range(n):
            temp = 0.0
            for k in range(n):
                temp += A[i][k] * B[k][j]
            C[i][j] = temp
    return C

def error_factorizacion(A, L, U):
    R = producto(L, U)
    n = len(A)
    err = 0.0
    for i in range(n):
        for j in range(n):
            diff = abs(A[i][j] - R[i][j])
            if diff > err:
                err = diff
    return err

# --------------------------------------------------------------
# PROGRAMA PRINCIPAL
# --------------------------------------------------------------
def main():
    archivo = input("Archivo CSV (A|b, coma, sin cabecera): ").strip()
    A, b = leer_csv(archivo)
    n = len(A)

    # Factorización
    L, U, m1, d1, s1 = factorizacion_LU_crout(A)

    # Sustituciones
    y, m2, d2, s2 = sustitucion_progresiva(L, b)
    x, m3, d3, s3 = sustitucion_regresiva(U, y)

    # Totales por fases
    mults_fact = m1
    divs_fact  = d1
    sums_fact  = s1

    mults_sust = m2 + m3
    divs_sust  = d2 + d3
    sums_sust  = s2 + s3

    # Totales globales
    mults_total = mults_fact + mults_sust
    divs_total  = divs_fact  + divs_sust
    sums_total  = sums_fact  + sums_sust
    total_ops   = mults_total + divs_total + sums_total

    # Costes teóricos
    coste_fact = (1/3) * n**3
    coste_sust = 2 * n**2
    coste_total = coste_fact + coste_sust

    # Resultados numéricos
    print("\nL =")
    for fila in L:
        print("  ", *[f"{v:12.6g}" for v in fila])
    print("\nU =")
    for fila in U:
        print("  ", *[f"{v:12.6g}" for v in fila])
    print("\ny =", " ".join(f"{v:12.6g}" for v in y))
    print("x =", " ".join(f"{v:12.6g}" for v in x))
    print(f"\nError máximo |A - L*U| = {error_factorizacion(A, L, U):.3e}")

    # Mostrar conteos
    print("\n=== COSTE COMPUTACIONAL (multiplicaciones/divisiones, sumas/restas) ===")
    # print("--- FACTORIZACIÓN ---")
    # print(f"Multiplicaciones/Divisiones: {mults_fact + divs_fact}")
    # print(f"Sumas/restas:                {sums_fact}")
    # print(f"Coste teórico (1/3 n^3):     {coste_fact:.0f}")
    # print(f"Relación real/teórico:       {(mults_fact + divs_fact + sums_fact)/coste_fact:.2f}")

    print("\n--- COSTE DE RESOLUCIÓN ---")
    print(f"Multiplicaciones/Divisiones: {mults_sust + divs_sust}")
    print(f"Sumas/restas:                {sums_sust}")
    print(f"Coste Real Total:            {mults_sust + divs_sust + sums_sust}")
    print(f"Coste teórico (2 n^2):       {coste_sust:.0f}")
    print(f"Relación real/teórico:       {(mults_sust + divs_sust + sums_sust)/coste_sust:.2f}")

    # print("\n--- TOTAL ---")
    # print(f"Total operaciones reales:    {total_ops}")
    # print(f"Multiplicaciones totales:    {mults_total}")
    # print(f"Divisiones totales:          {divs_total}")
    # print(f"Sumas/restas totales:        {sums_total}")
    # print(f"Coste teórico total:         {coste_total:.0f}")
    # print(f"Relación real/teórico total: {total_ops/coste_total:.2f}")

    # Guardar resultados
    pd.DataFrame([[f"{v:.5g}" for v in fila] for fila in L]).to_csv("L.csv", index=False, header=False)
    pd.DataFrame([[f"{v:.5g}" for v in fila] for fila in U]).to_csv("U.csv", index=False, header=False)
    pd.DataFrame([[f"{v:.5g}"] for v in x]).to_csv("x.csv", index=False, header=False)
    pd.DataFrame([[f"{v:.5g}"] for v in y]).to_csv("y.csv", index=False, header=False)
    print("\nArchivos guardados con formato .5g: L.csv, U.csv, x.csv, y.csv")

if __name__ == "__main__":
    main()
