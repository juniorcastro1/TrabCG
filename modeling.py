import numpy as np
from skimage.measure import marching_cubes

def extrair_malha(funcao_campo, limites=(-5, 5), resolucao=50):
    """
    Gera a grelha 3D e extrai vértices, faces e normais usando o algoritmo Marching Cubes.
    Realiza o mapeamento rigoroso dos índices da matriz para o espaço contínuo de coordenadas.
    """
    # 1. Cria a grelha do espaço escalar
    x, y, z = np.mgrid[
        limites[0]:limites[1]:complex(0, resolucao),
        limites[0]:limites[1]:complex(0, resolucao),
        limites[0]:limites[1]:complex(0, resolucao)
    ]
    
    # 2. Avalia a equação implícita F(x, y, z) em todos os pontos
    volume = funcao_campo(x, y, z)
    
    # 3. Extrai a geometria onde F(x, y, z) == 0.0 (superfície exata)
    vertices, faces, normals, values = marching_cubes(volume, level=0.0)
    
    # 4. Mapeamento Afim (Correção de Escala)
    # A API retorna índices (0 a 49). Precisamos converter de volta para o espaço [-5, 5]
    passo = (limites[1] - limites[0]) / (resolucao - 1)
    vertices = (vertices * passo) + limites[0]
    
    return vertices, faces, normals

# ==============================================================================
# EQUAÇÕES IMPLÍCITAS F(x, y, z) PARA AS 6 PEÇAS
# Valores negativos: Interior do sólido. Valores positivos: Exterior. Zero: Superfície.
# ==============================================================================

def gerar_campo_dama(x, y, z):
    """
    Peça de Dama: Um cilindro achatado.
    Equação do cilindro: x^2 + z^2 - R^2
    Limitado superiormente e inferiormente por planos.
    """
    raio = 3.5
    altura_meio = 0.5
    
    cilindro_infinito = x**2 + z**2 - raio**2
    plano_topo = y - altura_meio     # Corta tudo acima de y = 0.5
    plano_base = -y - altura_meio    # Corta tudo abaixo de y = -0.5
    
    # Intersecção: Máximo entre as funções
    dama = np.maximum(cilindro_infinito, np.maximum(plano_topo, plano_base))
    return dama

def gerar_campo_peao(x, y, z):
    """
    Peão: Base cónica truncada unida a uma esfera no topo com pescoço definido.
    """
    # Topo: Esfera de raio 1.2 centrada em y = 3.5
    esfera = x**2 + (y - 3.5)**2 + z**2 - 1.2**2

    # Base: Cone cuja espessura diminui conforme y cresce
    cone = x**2 + z**2 - (2.0 - 0.4 * y)**2

    # Limita o cone na altura do pescoço (y = 3.0)
    cone_limitado = np.maximum(cone, np.maximum(y - 3.0, -y))

    peao = np.minimum(esfera, cone_limitado)
    return peao

def gerar_campo_peao_elaborado(x, y, z):
    """
    Peão elaborado com 5 partes anatômicas:

    1. BASE     — cilindro largo e baixo (disco de apoio)
                  Equação do cilindro: x² + z² - r²
                  Limitado por dois planos horizontais com np.maximum.

    2. DEGRAU   — cilindro intermediário mais estreito, cria uma
                  transição escalonada entre base e barriga.
                  Mesma equação, raio e altura diferentes.

    3. BARRIGA  — elipsoide oblata (mais larga que alta).
                  Generaliza a esfera: cada eixo tem seu próprio raio.
                  Equação: (x²+z²)/a² + (y-cy)²/b² - 1
                  Com a > b o sólido achata-se horizontalmente.

    4. PESCOÇO  — cilindro fino entre barriga e cabeça.
                  Cria a estrangulação característica do peão.

    5. CABEÇA   — esfera perfeita no topo.
                  Equação: x² + (y-cy)² + z² - r²

    União final com np.minimum (ponto dentro de qualquer parte → dentro).
    """

    # ------------------------------------------------------------------
    # 1. BASE (disco)  — cilindro r=2.2, y: 0 → 0.7
    # ------------------------------------------------------------------
    base = np.maximum(
        x**2 + z**2 - 2.2**2,
        np.maximum(y - 0.7, -y)
    )

    # ------------------------------------------------------------------
    # 2. DEGRAU  — cilindro r=1.3, y: 0.5 → 1.4
    #    Sobrepõe levemente a base para garantir união contínua
    # ------------------------------------------------------------------
    degrau = np.maximum(
        x**2 + z**2 - 1.3**2,
        np.maximum(y - 1.4, -(y - 0.5))
    )

    # ------------------------------------------------------------------
    # 3. BARRIGA (elipsoide oblata)  — centrada em y=2.2
    #    a=1.6 (raio horizontal),  b=1.1 (semi-altura vertical)
    #    Equação geral do elipsoide:
    #        (x²+z²)/a²  +  (y-cy)²/b²  -  1  =  0
    #    Negativo dentro, positivo fora — mesma lógica da esfera,
    #    mas com escalas independentes por eixo.
    # ------------------------------------------------------------------
    a, b, cy_barriga = 1.6, 1.1, 2.2
    barriga = (x**2 + z**2) / a**2 + (y - cy_barriga)**2 / b**2 - 1.0

    # ------------------------------------------------------------------
    # 4. PESCOÇO  — cilindro fino r=0.58, y: 2.0 → 3.1
    #    Cria a estrangulação entre barriga e cabeça
    # ------------------------------------------------------------------
    pescoco = np.maximum(
        x**2 + z**2 - 0.58**2,
        np.maximum(y - 3.1, -(y - 2.0))
    )

    # ------------------------------------------------------------------
    # 5. CABEÇA  — esfera r=1.1 centrada em y=3.9
    # ------------------------------------------------------------------
    cabeca = x**2 + (y - 3.9)**2 + z**2 - 1.1**2

    # ------------------------------------------------------------------
    # UNIÃO FINAL (minimum = pertence a qualquer parte)
    # ------------------------------------------------------------------
    return np.minimum(
        base,
        np.minimum(degrau, np.minimum(barriga, np.minimum(pescoco, cabeca)))
    )

def gerar_campo_torre(x, y, z):
    """
    Torre: Cilindro grosso e perfeitamente reto.
    """
    raio = 2.0
    altura = 4.0

    cilindro = x**2 + z**2 - raio**2

    # Limita entre y = 0 e y = 4
    torre = np.maximum(cilindro, np.maximum(y - altura, -y))
    return torre

def gerar_campo_torre_elaborada(x, y, z):
    """
    Torre elaborada com 3 partes distintas:

    1. BASE: cilindro largo e baixo — ancora a peça visualmente.
       Interseção de cilindro (raio 2.8) com dois planos horizontais.

    2. CORPO: cilindro principal levemente cônico (raio varia de 2.0
       embaixo a 1.6 no topo), criando uma silhueta mais elegante.
       Cone implícito: x² + z² - (r0 - k*y)²

    3. PARAPEITO + AMEIAS: anel mais largo no topo (raio 2.4) do qual
       emergem 4 blocos retangulares (merlons) igualmente espaçados a 0°,
       90°, 180° e 270°.
       Cada bloco é a interseção de 3 pares de planos (uma caixa implícita):
           F_box = max(|x-cx|-hx, |y-cy|-hy, |z-cz|-hz)

    União entre partes = np.minimum (ponto dentro de qualquer parte → dentro).
    Interseção / recorte = np.maximum.
    """

    # ------------------------------------------------------------------
    # 1. BASE LARGA  (y: 0 → 0.8, raio 2.8)
    # ------------------------------------------------------------------
    base = np.maximum(
        x**2 + z**2 - 2.8**2,
        np.maximum(y - 0.8, -y)
    )

    # ------------------------------------------------------------------
    # 2. CORPO CÔNICO  (y: 0.6 → 3.8)
    # Raio decresce linearmente: r(y) = 2.0 - 0.10*(y - 0.6)
    # Cone implícito: x² + z² - r(y)² = 0
    # ------------------------------------------------------------------
    r_cone = 2.0 - 0.10 * (y - 0.6)
    corpo = np.maximum(
        x**2 + z**2 - r_cone**2,
        np.maximum(y - 3.8, -(y - 0.6))
    )

    # ------------------------------------------------------------------
    # 3. PARAPEITO (anel)  (y: 3.6 → 4.3, raio 2.4)
    # ------------------------------------------------------------------
    parapeito = np.maximum(
        x**2 + z**2 - 2.4**2,
        np.maximum(y - 4.3, -(y - 3.6))
    )

    # ------------------------------------------------------------------
    # 4. AMEIAS — 4 blocos (merlons)
    # Cada bloco é uma caixa centrada em (cx, yc, cz):
    #   F_box = max(|x-cx|-hw,  |y-yc|-hh,  |z-cz|-hw)
    # Raio do centro dos blocos: R=1.6; meia-largura hw=0.55; meia-altura hh=0.5
    # ------------------------------------------------------------------
    R  = 1.6
    hw = 0.55
    hh = 0.50
    yc = 4.80   # centro vertical dos blocos

    def merlon(cx, cz):
        return np.maximum(
            np.abs(x - cx) - hw,
            np.maximum(np.abs(y - yc) - hh, np.abs(z - cz) - hw)
        )

    ameias = np.minimum(
        np.minimum(merlon( R,  0), merlon(-R,  0)),
        np.minimum(merlon( 0,  R), merlon( 0, -R))
    )

    # ------------------------------------------------------------------
    # UNIÃO FINAL de todas as partes (minimum = união)
    # ------------------------------------------------------------------
    return np.minimum(base, np.minimum(corpo, np.minimum(parapeito, ameias)))

def gerar_campo_bispo(x, y, z):
    """
    Bispo: Cone alto e esbelto com uma pequena elipsoide/esfera no topo.
    """
    # Topo: Esfera menor (raio 1.0) em y = 4.5
    esfera = x**2 + (y - 4.5)**2 + z**2 - 1.0**2
    
    # Corpo: Cone longo
    cone = x**2 + z**2 - (2.0 - 0.25 * y)**2
    cone_limitado = np.maximum(cone, np.maximum(y - 4.0, -y))
    
    bispo = np.minimum(esfera, cone_limitado)
    return bispo

def gerar_campo_rainha(x, y, z):
    """
    Rainha: Corpo cónico muito alto, topo esférico com um adorno (coroa aproximada).
    """
    # Corpo principal
    cone = x**2 + z**2 - (2.2 - 0.15 * y)**2
    cone_limitado = np.maximum(cone, np.maximum(y - 6.0, -y))
    
    # Topo esférico (raio 1.2 em y = 6.5)
    esfera = x**2 + (y - 6.5)**2 + z**2 - 1.2**2
    
    rainha = np.minimum(esfera, cone_limitado)
    return rainha

def gerar_campo_rei(x, y, z):
    """
    Rei: A peça mais alta. Cone encorpado com uma cruz (aproximada por blocos) no topo.
    """
    # Corpo principal (mais alto que a Rainha)
    cone = x**2 + z**2 - (2.2 - 0.12 * y)**2
    cone_limitado = np.maximum(cone, np.maximum(y - 7.0, -y))
    
    # Cruz no topo: Dois paralelogramos (blocos) em y = 8.0
    # Bloco Vertical
    bloco_v_x = np.abs(x) - 0.3
    bloco_v_y = np.abs(y - 8.0) - 1.0
    bloco_v_z = np.abs(z) - 0.3
    cruz_v = np.maximum(bloco_v_x, np.maximum(bloco_v_y, bloco_v_z))
    
    # Bloco Horizontal
    bloco_h_x = np.abs(x) - 0.8
    bloco_h_y = np.abs(y - 8.0) - 0.3
    bloco_h_z = np.abs(z) - 0.3
    cruz_h = np.maximum(bloco_h_x, np.maximum(bloco_h_y, bloco_h_z))
    
    cruz = np.minimum(cruz_v, cruz_h)
    
    rei = np.minimum(cone_limitado, cruz)
    return rei