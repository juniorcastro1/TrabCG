import numpy as np

def rasterizar_bresenham(x1, y1, x2, y2):
    """
    Algoritmo de Bresenham operando estritamente com inteiros.
    Elimina a necessidade de divisões de ponto flutuante do DDA.
    """
    pontos = []
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Determina a direção (sinal) do incremento
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    # Variável de erro (decisão de avanço)
    err = dx - dy

    while True:
        pontos.append((x1, y1))
        
        # Condição de parada rigorosa ao atingir o vértice destino
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        
        # Avanço no eixo X
        if e2 > -dy:
            err -= dy
            x1 += sx
            
        # Avanço no eixo Y
        if e2 < dx:
            err += dx
            y1 += sy

    return pontos

def calcular_area_triangulo(xa, ya, xb, yb, xc, yc):
    """
    Cálculo exato da área usando o produto vetorial 2D (determinante).
    """
    return 0.5 * abs((xa * (yb - yc)) + (xb * (yc - ya)) + (xc * (ya - yb)))

def coordenadas_baricentricas(xp, yp, xa, ya, xb, yb, xc, yc):
    """
    Calcula as coordenadas alpha, beta e gamma para interpolação de cores e profundidade (Z).
    """
    area_total = calcular_area_triangulo(xa, ya, xb, yb, xc, yc)
    
    if area_total == 0:
        return -1, -1, -1 # Triângulo degenerado (linha)
        
    area_alpha = calcular_area_triangulo(xp, yp, xb, yb, xc, yc)
    area_beta  = calcular_area_triangulo(xa, ya, xp, yp, xc, yc)
    
    alpha = area_alpha / area_total
    beta  = area_beta / area_total
    gamma = 1.0 - alpha - beta
    
    return alpha, beta, gamma

def scan_line_par_impar(vertices_2d, largura_tela=800, altura_tela=800):
    """
    Algoritmo Par-Ímpar (Scan Line) minucioso com clamping de bordas para
    evitar processamento ao infinito em projeções distorcidas.
    """
    pixels_internos = []
    
    y_min = int(np.floor(min(v[1] for v in vertices_2d)))
    y_max = int(np.ceil(max(v[1] for v in vertices_2d)))
    
    # Proteção estrita contra coordenadas fora da tela
    y_min = max(0, y_min)
    y_max = min(altura_tela - 1, y_max)
    
    num_vertices = len(vertices_2d)
    
    for y in range(y_min, y_max + 1):
        intersecoes = []
        for i in range(num_vertices):
            p1 = vertices_2d[i]
            p2 = vertices_2d[(i + 1) % num_vertices]
            y1, y2 = p1[1], p2[1]
            x1, x2 = p1[0], p2[0]
            
            # Avalia interseção analítica da reta horizontal com a aresta
            if (y1 <= y < y2) or (y2 <= y < y1):
                # Equação da reta interpolada para achar o X
                x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                intersecoes.append(x_int)
        
        intersecoes.sort()
        
        # Lógica Par-Ímpar: Pinta entre pares de interseções
        for i in range(0, len(intersecoes), 2):
            if i + 1 < len(intersecoes):
                x_inicio = int(np.ceil(intersecoes[i]))
                x_fim = int(np.floor(intersecoes[i+1]))
                
                x_inicio = max(0, x_inicio)
                x_fim = min(largura_tela - 1, x_fim)
                
                for x in range(x_inicio, x_fim + 1):
                    pixels_internos.append((x, y))
                    
    return pixels_internos