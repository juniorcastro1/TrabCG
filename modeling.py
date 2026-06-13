import numpy as np
from skimage.measure import marching_cubes
from primitives import (
    cilindro, cone_truncado, esfera, elipsoide, caixa, toro,
    uniao, intersecao, subtracao, uniao_suave
)


def extrair_malha(funcao_campo, limites=(-5, 12), resolucao=100):
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
    """Dama: disco tipo moeda com borda saliente e 3 ranhuras concêntricas."""
    RAIO   = 2.3
    ALTURA = 0.9
    GROOVE = 0.18  # profundidade de cada ranhura (r_tubo do toro)

    # Corpo: disco principal + anel lateral que forma a borda saliente
    disco = cilindro(x, y, z, raio=RAIO,       y_min=0.0,         y_max=ALTURA)
    borda = cilindro(x, y, z, raio=RAIO + 0.2, y_min=ALTURA*0.15, y_max=ALTURA*0.85)
    corpo = uniao(disco, borda)

    # 3 ranhuras no topo (toros centrados na face superior, cy=ALTURA)
    topo_1 = toro(x, y, z, cx=0, cy=ALTURA, cz=0, r_maior=0.50, r_tubo=GROOVE)
    topo_2 = toro(x, y, z, cx=0, cy=ALTURA, cz=0, r_maior=1.15, r_tubo=GROOVE)
    topo_3 = toro(x, y, z, cx=0, cy=ALTURA, cz=0, r_maior=1.80, r_tubo=GROOVE)

    # 3 ranhuras na base (espelhadas em cy=0.0)
    base_1 = toro(x, y, z, cx=0, cy=0.0, cz=0, r_maior=0.50, r_tubo=GROOVE)
    base_2 = toro(x, y, z, cx=0, cy=0.0, cz=0, r_maior=1.15, r_tubo=GROOVE)
    base_3 = toro(x, y, z, cx=0, cy=0.0, cz=0, r_maior=1.80, r_tubo=GROOVE)

    return subtracao(corpo, uniao(topo_1, topo_2, topo_3, base_1, base_2, base_3))


def gerar_campo_peao(x, y, z):

    base = cilindro(x, y, z,  raio=2, y_min=0.0, y_max=0.7)
    corpo = cone_truncado(x, y, z, y_min=0.6, y_max=3.6, r_base=1.3, r_topo=0.3)
    anel = cilindro(x, y, z, raio=1.3, y_min=3.0, y_max=3.5)
    topo = esfera(x, y, z, cx=0, cy=4.2, cz=0, raio=0.9)

    return uniao(
        base, corpo, anel, topo
    )

def gerar_campo_torre(x, y, z):
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
    """
    Rainha com perfil de vaso/taça:

    CORPO (reformulado):
      - Base disc larga
      - Barriga arredondada via ELIPSOIDE (bojuda, não cônica)
      - Afunilamento via CONE TRUNCADO do meio para a cintura
      - UNIAO_SUAVE entre barriga e afunilamento elimina a aresta de junção
      - Pescoço muito fino
      - Aro da coroa (cilindro)

    TOPO (mantido):
      - 6 esferas-pontas em círculo (a cada 60°)
      - Orbe central
    """
    # --- Coroa: mantida igual ---
    r_coroa = 1.15
    y_coroa = 6.05
    r_ponta = 0.36

    pontas_coroa = uniao(*[
        esfera(x, y, z,
               cx=r_coroa * np.cos(np.radians(ang)),
               cy=y_coroa,
               cz=r_coroa * np.sin(np.radians(ang)),
               raio=r_ponta)
        for ang in range(0, 360, 60)
    ])

    # Elipsoide bojudo na parte baixa: a=1.85 (horizontal), b=1.9 (vertical)
    # centrado em y=1.2, cobre aproximadamente y ∈ [-0.7, 3.1]
    barriga = elipsoide(x, y, z, cx=0, cy=1.9, cz=0, a=1.85, b=1.9)

    # Cone que afunila do meio (r=1.35) até a cintura (r=0.52)
    afunilamento = cone_truncado(x, y, z, y_min=2.6, y_max=4.9, r_base=1.35, r_topo=0.52)

    # Smooth union funde as duas formas sem aresta visível na junção
    corpo = uniao_suave(barriga, afunilamento, k=0.7)

    return uniao(
        cilindro(x, y, z, raio=2.2, y_min=0.0, y_max=0.55),   # base
        corpo,                                                    # barriga + afunilamento
        cilindro(x, y, z, raio=0.52, y_min=4.3, y_max=5.0),   # pescoço fino
        cilindro(x, y, z, raio=1.55, y_min=4.8, y_max=5.8),   # aro da coroa
        pontas_coroa, # 6 pontas
        cilindro(x, y,z, raio=0.3, y_min=5.8, y_max=6.55),
        esfera(x, y, z, cx=0, cy=6.55, cz=0, raio=0.52), # orbe
    )


def gerar_campo_rei(x, y, z):
    """
    Rei com perfil de vaso (mesmo padrão da Rainha, ligeiramente mais robusto)
    e cruz no topo.

    CORPO:
      - Base disc larga
      - Barriga elipsoide (a=1.95, b=2.0 — mais encorpado que a Rainha)
      - Afunilamento cônico até a cintura
      - uniao_suave para fundir barriga e afunilamento
      - Pescoço fino
      - Colar (aro antes da cruz)
      - Haste curta conectando colar à cruz

    TOPO:
      - Cruz: barra vertical + barra horizontal (dois blocos ortogonais)
        centrados no mesmo ponto, a cruz resulta da união das caixas
    """
    # --- Cruz ---
    cruz = uniao(
        caixa(x, y, z, cx=0, cy=7.2, cz=0, hx=0.32, hy=1.1, hz=0.32),  # barra vertical
        caixa(x, y, z, cx=0, cy=7.2, cz=0, hx=0.90, hy=0.32, hz=0.32),  # barra horizontal
    )

    # --- Corpo estilo vaso (mais robusto que a Rainha) ---
    barriga      = elipsoide(x, y, z, cx=0, cy=2, cz=0, a=1.95, b=2.0)
    afunilamento = cone_truncado(x, y, z, y_min=2.6, y_max=5.2, r_base=1.45, r_topo=0.60)
    corpo        = uniao_suave(barriga, afunilamento, k=0.7)

    return uniao(
        cilindro(x, y, z, raio=2.5,  y_min=0.0, y_max=0.60),  # base
        corpo,                                                    # barriga + afunilamento
        cilindro(x, y, z, raio=0.60, y_min=4.8, y_max=5.65),  # pescoço fino
        cilindro(x, y, z, raio=1.45, y_min=5.3, y_max=6.1),   # colar
        cilindro(x, y, z, raio=0.40, y_min=6.0, y_max=6.35),  # haste até a cruz
        cruz,
    )
