"""
primitives.py — Biblioteca de formas implícitas e operações booleanas
======================================================================
Cada função retorna um campo escalar F(x,y,z) onde:
    F < 0  →  interior do sólido
    F = 0  →  superfície
    F > 0  →  exterior

Os parâmetros x, y, z são arrays NumPy (grades 3D vindas de np.mgrid),
portanto todas as operações são vetorizadas automaticamente.
"""

import numpy as np


# ======================================================================
# OPERAÇÕES BOOLEANAS
# ======================================================================

def uniao(*formas):
    """
    União de N formas: ponto está dentro se estiver em QUALQUER forma.
    Implementação: mínimo das funções de campo.
    Equivale ao OR lógico sobre o sinal de F.
    """
    resultado = formas[0]
    for f in formas[1:]:
        resultado = np.minimum(resultado, f)
    return resultado


def intersecao(*formas):
    """
    Interseção de N formas: ponto está dentro apenas se estiver em TODAS.
    Implementação: máximo das funções de campo.
    Equivale ao AND lógico sobre o sinal de F.
    """
    resultado = formas[0]
    for f in formas[1:]:
        resultado = np.maximum(resultado, f)
    return resultado


def subtracao(a, b):
    """
    Subtração: retira a forma B do sólido A.
    Ponto está dentro se estiver em A E fora de B.
    Implementação: max(A, -B)  — inverte o sinal de B para "fora vira dentro".
    """
    return np.maximum(a, -b)


def uniao_suave(a, b, k=1.0):
    """
    União com transição suave (smooth union).
    Elimina a aresta viva que np.minimum cria na junção entre duas formas.
    k controla o raio de blend: k pequeno ≈ aresta viva, k grande ≈ fusão ampla.

    Fórmula (Inigo Quilez):
        h = max(k - |A - B|, 0) / k
        F = min(A, B) - h²·k/4
    """
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k / 4.0


# ======================================================================
# PRIMITIVAS — FORMAS BÁSICAS
# ======================================================================

def cilindro(x, y, z, raio, y_min, y_max):
    """
    Cilindro reto com eixo em Y, entre y_min e y_max.

    Equação radial:  x² + z² - r²
      < 0 dentro do tubo, > 0 fora.

    Dois planos horizontais limitam a altura:
      plano superior: y - y_max  (negativo abaixo de y_max)
      plano inferior: -(y - y_min)  (negativo acima de y_min)

    Interseção dos três = cilindro finito.
    """
    return intersecao(
        x**2 + z**2 - raio**2,
        y - y_max,
        -(y - y_min)
    )


def cone_truncado(x, y, z, y_min, y_max, r_base, r_topo):
    """
    Cone truncado (frustum): raio varia linearmente entre r_base (em y_min)
    e r_topo (em y_max).

    O raio em função de y é interpolado linearmente:
        t = (y - y_min) / (y_max - y_min)   ∈ [0, 1]
        r(y) = r_base + t·(r_topo - r_base)
             = r_base·(1-t) + r_topo·t

    Superfície implícita: x² + z² - r(y)²  =  0

    Casos especiais:
      r_topo = r_base  →  cilindro reto
      r_topo = 0       →  cone pontudo
    """
    h = y_max - y_min
    t = (y - y_min) / h
    r = r_base + t * (r_topo - r_base)
    return intersecao(
        x**2 + z**2 - r**2,
        y - y_max,
        -(y - y_min)
    )


def esfera(x, y, z, cx, cy, cz, raio):
    """
    Esfera centrada em (cx, cy, cz) com raio dado.

    Deriva diretamente do Teorema de Pitágoras em 3D:
    distância ao centro = sqrt((x-cx)²+(y-cy)²+(z-cz)²)
    Superfície onde distância = raio:
        (x-cx)² + (y-cy)² + (z-cz)² - raio²  =  0
    """
    return (x - cx)**2 + (y - cy)**2 + (z - cz)**2 - raio**2


def elipsoide(x, y, z, cx, cy, cz, a, b):
    """
    Elipsoide com simetria rotacional em torno do eixo Y.
    a = raio horizontal (X e Z), b = semialtura (Y).

    Generaliza a esfera dividindo cada eixo pelo seu próprio raio:
        (x-cx)²/a²  +  (y-cy)²/b²  +  (z-cz)²/a²  -  1  =  0

    a > b  →  oblato  (disco achatado, como a Terra)
    b > a  →  prolato (forma de ovo ou limão)
    a = b  →  esfera  (caso degenerado)
    """
    return (x - cx)**2 / a**2 + (y - cy)**2 / b**2 + (z - cz)**2 / a**2 - 1.0


def caixa(x, y, z, cx, cy, cz, hx, hy, hz):
    """
    Caixa (paralelepípedo) centrada em (cx, cy, cz) com meias-dimensões hx, hy, hz.

    Para cada eixo, |x - cx| - hx  é negativo dentro da faixa [cx-hx, cx+hx]
    e positivo fora. O valor absoluto cria as duas paredes opostas ao mesmo tempo.

    Interseção dos três eixos = interior da caixa:
        max(|x-cx|-hx,  |y-cy|-hy,  |z-cz|-hz)
    """
    return intersecao(
        np.abs(x - cx) - hx,
        np.abs(y - cy) - hy,
        np.abs(z - cz) - hz
    )


def toro(x, y, z, cx, cy, cz, r_maior, r_tubo):
    """
    Toro (forma de rosca/donut) centrado em (cx, cy, cz), deitado no plano XZ.

    R_maior = distância do centro do toro ao centro do tubo circular
    R_tubo = raio da secção transversal do tubo

    Derivação: a distância de um ponto ao círculo central do toro é:
        d_anel = sqrt(x²+z²) - r_maior (distância ao anel no plano XZ)
    A superfície do toro é onde a distância 3D ao anel = r_tubo:
        sqrt(d_anel² + y²) - r_tubo = 0
    Elevando ao quadrado para remover a raiz externa:
        (sqrt(x²+z²) - r_maior)² + y² - r_tubo² = 0
    """
    d_anel = np.sqrt((x - cx)**2 + (z - cz)**2) - r_maior
    return d_anel**2 + (y - cy)**2 - r_tubo**2

