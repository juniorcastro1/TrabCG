import numpy as np

def normalizar(v):
    norma = np.linalg.norm(v)
    if norma == 0:
        return v
    return v / norma

def calcular_iluminacao_phong(vertice_mundo, normal, posicao_luz, posicao_camera, 
                              cor_objeto, coeficientes=(0.2, 0.7, 0.5), brilho=32):
    """
    Cálculo vetorial minucioso do Modelo de Phong.
    coeficientes: (ka, kd, ks) correspondentes a ambiente, difusa e especular.
    """
    ka, kd, ks = coeficientes
    cor_luz = np.array([1.0, 1.0, 1.0]) # Luz branca
    cor_obj = np.array(cor_objeto)
    
    # 1. Vetores Unitários Fundamentais
    N = normalizar(normal)
    L = normalizar(posicao_luz - vertice_mundo)
    V = normalizar(posicao_camera - vertice_mundo)
    
    # 2. Componente Ambiente
    ambiente = ka * cor_luz * cor_obj
    
    # 3. Componente Difusa (Lei do Cosseno de Lambert)
    # Produto escalar entre N e L. Se a luz vem por trás (negativo), a difusa é 0.
    l_dot_n = max(np.dot(L, N), 0.0)
    difusa = kd * l_dot_n * cor_luz * cor_obj
    
    # 4. Componente Especular
    especular = np.array([0.0, 0.0, 0.0])
    if l_dot_n > 0:
        # Cálculo estrito do vetor de reflexão R
        R = normalizar(2.0 * l_dot_n * N - L)
        r_dot_v = max(np.dot(R, V), 0.0)
        especular = ks * (r_dot_v ** brilho) * cor_luz
        
    # 5. Somatório Final
    cor_final = ambiente + difusa + especular
    
    # Garante que os valores RGB permaneçam no limite [0, 1]
    return np.clip(cor_final, 0.0, 1.0)