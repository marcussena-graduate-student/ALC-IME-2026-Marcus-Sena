"""Testes da função resolve_lu (alc-p1-Q5-Marcus.py).
numpy.linalg/allclose são usados SOMENTE aqui, para conferência."""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("q5", "alc-p1-Q5-Marcus.py")
q5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q5)
resolve_lu = q5.resolve_lu

np.set_printoptions(precision=4, suppress=True)


def verifica(nome, A, b):
    print(f"\n=== {nome} ===")
    L, U, x = resolve_lu(A, b)
    A, b = np.array(A, float), np.array(b, float)
    print("L =\n", L)
    print("U =\n", U)
    print("x =", x)
    ok = (np.allclose(L @ U, A)
          and np.allclose(A @ x, b)
          and np.allclose(x, np.linalg.solve(A, b))
          and np.allclose(np.diag(L), 1)
          and np.allclose(np.triu(L, 1), 0)
          and np.allclose(np.tril(U, -1), 0))
    print("LU = A, Ax = b, formatos de L e U:", "OK" if ok else "FALHOU")
    return ok


resultados = []

# 1) Exemplo clássico 3x3 (solução esperada x = [1, 1, 2])
resultados.append(verifica("3x3 clássico",
                           [[2, 1, 1], [4, -6, 0], [-2, 7, 2]], [5, -2, 9]))

# 2) 2x2 simples (x = [1, 2])
resultados.append(verifica("2x2", [[4, 3], [6, 3]], [10, 12]))

# 3) 4x4 com solução conhecida x = [1, -1, 2, 0.5]
A4 = [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]]
x4 = np.array([1, -1, 2, 0.5])
resultados.append(verifica("4x4 (diag. dominante)", A4, np.array(A4) @ x4))

# 4) Matriz aleatória 6x6 diagonalmente dominante
rng = np.random.default_rng(42)
A6 = rng.uniform(-1, 1, (6, 6)) + 6 * np.eye(6)
resultados.append(verifica("6x6 aleatória", A6, rng.uniform(-5, 5, 6)))

# 5) Matriz 1x1
resultados.append(verifica("1x1", [[5]], [10]))

# 6) Pivô nulo logo no início -> deve lançar Exception
print("\n=== Pivô nulo (A[0][0] = 0) ===")
try:
    resolve_lu([[0, 1], [1, 1]], [1, 2])
    print("FALHOU: nenhuma exceção lançada"); resultados.append(False)
except Exception as e:
    print("Exceção lançada corretamente:\n ", e); resultados.append(True)

# 7) Pivô nulo que surge no meio da eliminação -> Exception
print("\n=== Pivô nulo surgindo na eliminação ===")
try:
    resolve_lu([[1, 2, 3], [2, 4, 5], [7, 8, 9]], [1, 2, 3])
    print("FALHOU: nenhuma exceção lançada"); resultados.append(False)
except Exception as e:
    print("Exceção lançada corretamente:\n ", e); resultados.append(True)

print(f"\nResumo: {sum(resultados)}/{len(resultados)} testes passaram.")
