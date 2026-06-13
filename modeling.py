import numpy as np
from skimage.measure import marching_cubes
from primitives import (
    cilindro, cone_truncado, esfera, elipsoide, caixa, toro, capsula,
    uniao, intersecao, subtracao, uniao_suave
)


def extrair_malha(funcao_campo, limites=(-5, 12), resolucao=80):
    """
    Gera a grelha 3D e extrai vértices, faces e normais usando Marching Cubes.
    """
    x, y, z = np.mgrid[
        limites[0]:limites[1]:complex(0, resolucao),
        limites[0]:limites[1]:complex(0, resolucao),
        limites[0]:limites[1]:complex(0, resolucao)
    ]
    volume = funcao_campo(x, y, z)
    vertices, faces, normals, values = marching_cubes(volume, level=0.0)
    passo = (limites[1] - limites[0]) / (resolucao - 1)
    vertices = (vertices * passo) + limites[0]
    return vertices, faces, normals


# ==============================================================================
# PEÇAS — usando primitives.py
# ==============================================================================

def gerar_campo_dama(x, y, z):
    """Dama: disco achatado (cilindro de baixa altura)."""
    return cilindro(x, y, z, raio=3.5, y_min=-0.5, y_max=0.5)


def gerar_campo_peao(x, y, z):

    base = cilindro(x, y, z,  raio=2, y_min=0.0, y_max=0.7)
    corpo = cone_truncado(x, y, z, y_min=0.6, y_max=3.6, r_base=1.5, r_topo=0.5)
    anel = cilindro(x, y, z, raio=1.5, y_min=3.0, y_max=3.5)
    topo = esfera(x, y, z, cx=0, cy=4.2, cz=0, raio=1)

    return uniao(
        base, corpo, anel, topo
    )

def gerar_campo_torre(x, y, z):
    """Torre simples: cilindro reto."""
    return cilindro(x, y, z, raio=2.0, y_min=0.0, y_max=4.0)


def gerar_campo_torre_elaborada(x, y, z):
    """
    Torre elaborada: construída por camadas simples, seguindo a mesma lógica
    usada no peão e no bispo: base larga, corpo, anéis, topo e ameias.
    """

    base_1 = cilindro(x, y, z, raio=2.3, y_min=0.0, y_max=0.5)
    corpo_1 = cone_truncado(x, y, z, y_min=0.5, y_max=2.0, r_base=1.6, r_topo=1.1)
    corpo_2 = cilindro(x, y, z, raio=1.1, y_min=0.4, y_max=3.5)

    anel_topo = cilindro(x, y, z, raio=1.75, y_min=3.55, y_max=4.3)

    miolo_topo = cilindro(x, y, z, raio=1.15, y_min=3.8, y_max=3.0)

    ameia_frente = caixa(x, y, z, cx=0.0, cy=4.55, cz=1.15, hx=0.45, hy=0.45, hz=0.45)
    ameia_tras = caixa(x, y, z, cx=0.0, cy=4.55, cz=-1.15, hx=0.45, hy=0.45, hz=0.45)
    ameia_direita = caixa(x, y, z, cx=1.15, cy=4.55, cz=0.0, hx=0.45, hy=0.45, hz=0.45)
    ameia_esquerda = caixa(x, y, z, cx=-1.15, cy=4.55, cz=0.0, hx=0.45, hy=0.45, hz=0.45)

    abertura = cilindro(x, y, z, raio=0.65, y_min=3.05, y_max=5)

    topo_com_ameias = subtracao(
        uniao(
            miolo_topo,
            ameia_frente,
            ameia_tras,
            ameia_direita,
            ameia_esquerda,
        ),
        abertura
    )

    return uniao(
        base_1,
        corpo_1,
        corpo_2,
        anel_topo,
        miolo_topo,
        topo_com_ameias,
    )


def gerar_campo_bispo(x, y, z):
    """Bispo: cone esbelto com esfera no topo."""

    base = cilindro(x, y, z,  raio=2, y_min=0.0, y_max=0.7)
    corpo_1 = cone_truncado(x, y, z, y_min=0.6, y_max=1.6, r_base=1.5, r_topo=0.8)
    corpo_2 = cilindro(x, y, z, raio=0.8, y_min=1.5, y_max=3.0)
    anel_1 = cilindro(x, y, z, raio=1.2, y_min=3.0, y_max=3.5)
    anel_2 = cilindro(x, y, z, raio=1.5, y_min=3.5, y_max=4.0)
    topo_1 = esfera(x, y, z, cx=0, cy=4.6, cz=0, raio=1)
    topo_2 = esfera(x, y, z, cx=0, cy=6.0, cz=0, raio=0.5)

    return uniao(
        base, corpo_1, corpo_2, anel_1, anel_2, topo_1, topo_2
    )


def gerar_campo_rainha(x, y, z):
    """Rainha: cone alto com esfera no topo."""
    return uniao(
        cone_truncado(x, y, z, y_min=0.0, y_max=6.0, r_base=2.2, r_topo=0.6),
        esfera(x, y, z, cx=0, cy=6.5, cz=0, raio=1.2),
    )


def gerar_campo_rei(x, y, z):
    """Rei: cone encorpado com cruz no topo (dois blocos ortogonais)."""
    cruz = uniao(
        caixa(x, y, z, cx=0, cy=8.0, cz=0, hx=0.3, hy=1.0, hz=0.3),  # vertical
        caixa(x, y, z, cx=0, cy=8.0, cz=0, hx=0.8, hy=0.3, hz=0.3),  # horizontal
    )
    return uniao(
        cone_truncado(x, y, z, y_min=0.0, y_max=7.0, r_base=2.2, r_topo=0.5),
        cruz,
    )
