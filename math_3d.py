import numpy as np

def translacao(tx, ty, tz):
    """ Matriz de Translação 4x4 """
    return np.array([
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0]
    ])

def escala(sx, sy, sz):
    """ Matriz de Escala 4x4 """
    return np.array([
        [sx,  0.0, 0.0, 0.0],
        [0.0, sy,  0.0, 0.0],
        [0.0, 0.0, sz,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotacao_y(theta_radianos):
    """ Matriz de Rotação em torno do eixo Y 4x4 """
    cos_t = np.cos(theta_radianos)
    sin_t = np.sin(theta_radianos)
    return np.array([
        [ cos_t, 0.0, sin_t, 0.0],
        [   0.0, 1.0,   0.0, 0.0],
        [-sin_t, 0.0, cos_t, 0.0],
        [   0.0, 0.0,   0.0, 1.0]
    ])

def look_at(eye, at, up):
    """
    Cálculo meticuloso da base vetorial do sistema de coordenadas da câmera.
    """
    eye = np.array(eye, dtype=float)
    at = np.array(at, dtype=float)
    up = np.array(up, dtype=float)

    # 1. Vetor n (Eixo Z da câmera - aponta de 'at' para 'eye')
    n = eye - at
    n = n / np.linalg.norm(n)

    # 2. Vetor u (Eixo X da câmera - ortogonal a 'up' e 'n')
    u = np.cross(up, n)
    u = u / np.linalg.norm(u)

    # 3. Vetor v (Eixo Y da câmera - ortogonal a 'n' e 'u')
    v = np.cross(n, u)

    # 4. Matriz de Visualização (View Matrix) = Alinhamento de base * Translação
    view_matrix = np.array([
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [v[0], v[1], v[2], -np.dot(v, eye)],
        [n[0], n[1], n[2], -np.dot(n, eye)],
        [ 0.0,  0.0,  0.0,               1.0]
    ])
    
    return view_matrix

def projecao_perspectiva(fov_graus, aspecto, z_near, z_far):
    """
    Matriz de Projeção em Perspectiva 4x4 estrita.
    """
    fov_rad = np.radians(fov_graus)
    f = 1.0 / np.tan(fov_rad / 2.0)
    
    matriz = np.zeros((4, 4), dtype=float)
    matriz[0, 0] = f / aspecto
    matriz[1, 1] = f
    
    # Cálculo minucioso das componentes de profundidade (Z)
    matriz[2, 2] = (z_far + z_near) / (z_near - z_far)
    matriz[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    matriz[3, 2] = -1.0 # Guarda o Z original na componente W para a divisão perspectiva
    
    return matriz

def divisao_perspectiva(vertice_4d):
    """ Divide x, y, z pela coordenada homogênea W """
    if vertice_4d[3] != 0:
        return vertice_4d[:3] / vertice_4d[3]
    return vertice_4d[:3]

def normalizar(v):
    """Retorna o vetor unitário (norma igual a 1)"""
    norma = np.linalg.norm(v)
    if norma == 0:
        return v
    return v / norma