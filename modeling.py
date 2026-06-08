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