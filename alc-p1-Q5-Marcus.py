"""
Prova 1 - Álgebra Linear Computacional - Pós-Grad - 2026.2
5ª Questão - Solução de Ax = b via decomposição LU (sem pivoteamento)

Aluno: Marcus Sena

Uso de numpy restrito à criação de arrays (numpy.array, numpy.eye,
numpy.zeros, numpy.ones), conforme o enunciado. Todas as operações de
decomposição e substituição são implementadas manualmente com laços.
"""

import numpy

# Tolerância para considerar um pivô como nulo (evita divisão por valores
# numericamente iguais a zero, resultantes de erros de arredondamento).
TOL_PIVO = 1e-12


def decomposicao_lu(A):
    """
    Constrói explicitamente L (triangular inferior com diagonal unitária)
    e U (triangular superior) tais que A = LU, por eliminação de Gauss
    SEM pivoteamento.

    Os elementos de L abaixo da diagonal são os multiplicadores usados
    na eliminação: l_ik = u_ik / u_kk.
    """
    n = len(A)

    # U começa como cópia de A (em ponto flutuante); L começa como identidade
    U = numpy.array(A, dtype=float)
    L = numpy.eye(n)

    for k in range(n):
        pivo = U[k][k]
        if abs(pivo) < TOL_PIVO:
            raise Exception(
                f"Pivô nulo encontrado na posição ({k}, {k}). A decomposição LU "
                "sem pivoteamento não pode prosseguir. Utilize uma função "
                "alternativa para resolver o sistema (por exemplo, LU com "
                "pivoteamento parcial, PA = LU)."
            )

        # Elimina os elementos abaixo do pivô na coluna k
        for i in range(k + 1, n):
            m = U[i][k] / pivo        # multiplicador
            L[i][k] = m               # o multiplicador é armazenado em L
            for j in range(k, n):
                U[i][j] = U[i][j] - m * U[k][j]
            U[i][k] = 0.0             # zera explicitamente (evita resíduo numérico)

    return L, U


def substituicao_progressiva(L, b):
    """Resolve Ly = b, com L triangular inferior, de cima para baixo."""
    n = len(b)
    y = numpy.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i][j] * y[j]
        y[i] = (b[i] - soma) / L[i][i]   # L[i][i] = 1
    return y


def substituicao_regressiva(U, y):
    """Resolve Ux = y, com U triangular superior, de baixo para cima."""
    n = len(y)
    x = numpy.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i][j] * x[j]
        x[i] = (y[i] - soma) / U[i][i]
    return x


def resolve_lu(A, b):
    """
    Resolve o sistema linear Ax = b por decomposição LU (A = LU).

    Parâmetros
    ----------
    A : matriz quadrada n x n (lista de listas ou numpy array)
    b : vetor de n elementos

    Retorna
    -------
    L : matriz triangular inferior com diagonal unitária
    U : matriz triangular superior
    x : vetor solução de Ax = b

    Lança Exception se for encontrado um pivô nulo.
    """
    A = numpy.array(A, dtype=float)
    b = numpy.array(b, dtype=float)

    # Validações de dimensão
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada (n x n).")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("O vetor b deve ter n elementos, compatível com A.")

    L, U = decomposicao_lu(A)
    y = substituicao_progressiva(L, b)   # Ly = b
    x = substituicao_regressiva(U, y)    # Ux = y

    return L, U, x


if __name__ == "__main__":
    # Exemplo de uso
    A = numpy.array([[2, 1, 1],
                     [4, -6, 0],
                     [-2, 7, 2]], dtype=float)
    b = numpy.array([5, -2, 9], dtype=float)

    print("A =\n", A)
    print("b =", b)

    L, U, x = resolve_lu(A, b)
    print("L =\n", L)
    print("U =\n", U)
    print("x =", x)
