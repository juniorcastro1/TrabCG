import numpy as np
import modeling
import math_3d
import illumination
import rasterizer
import matplotlib.pyplot as plt

def main():
    print("Iniciando o Pipeline Gráfico...")
    
    # ---------------------------------------------------------
    # FASE 1: Matrizes da Câmera e Luz
    # ---------------------------------------------------------
    posicao_camera = np.array([0.0, 5.0, 15.0])
    posicao_luz = np.array([10.0, 15.0, 10.0]) # Subi a luz para iluminar melhor a cabeça
    
    view_matrix = math_3d.look_at(eye=posicao_camera, at=[0.0, 1.0, 0.0], up=[0.0, 1.0, 0.0])
    proj_matrix = math_3d.projecao_perspectiva(fov_graus=60, aspecto=1.0, z_near=0.1, z_far=100)
    
    # ---------------------------------------------------------
    # FASE 2: Modelagem e Matriz do Mundo
    # ---------------------------------------------------------
    print("A extrair a malha do Peão (Marching Cubes)...")
    vert_peao, faces_peao, norm_peao = modeling.extrair_malha(modeling.gerar_campo_peao)
    
    m_escala = math_3d.escala(1.5, 1.5, 1.5)
    m_rotacao = math_3d.rotacao_y(0.0)
    m_translacao = math_3d.translacao(0.0, -2.0, 0.0)
    matriz_mundo_peao = np.dot(m_translacao, np.dot(m_rotacao, m_escala))
    
    # ---------------------------------------------------------
    # FASE 3 e 4: Processamento Vetorial
    # ---------------------------------------------------------
    print("A calcular vértices e projeções...")
    vertices_tela = []
    vertices_mundo = []
    profundidades_z = [] # Guardaremos o Z de cada vértice para o Z-Buffer
    
    for i in range(len(vert_peao)):
        v_homogeneo = np.array([vert_peao[i][0], vert_peao[i][1], vert_peao[i][2], 1.0])
        v_mundo = np.dot(matriz_mundo_peao, v_homogeneo)
        vertices_mundo.append(v_mundo[:3])
        
        v_camera = np.dot(view_matrix, v_mundo)
        v_proj = np.dot(proj_matrix, v_camera)
        v_ndc = math_3d.divisao_perspectiva(v_proj)
        
        profundidades_z.append(v_ndc[2]) # Z em Normalized Device Coordinates
        
        x_tela = (v_ndc[0] + 1.0) * 0.5 * 800
        y_tela = (1.0 - v_ndc[1]) * 0.5 * 800
        vertices_tela.append((x_tela, y_tela))

    # ---------------------------------------------------------
    # FASE 5: Back-face Culling, Z-Buffer e Baricêntrica
    # ---------------------------------------------------------
    imagem = np.zeros((800, 800, 3), dtype=np.uint8)
    
    # Cria a matriz do Z-Buffer inicializada com infinito
    z_buffer = np.full((800, 800), np.inf)
    
    print("A renderizar com Z-Buffer e Interpolação Baricêntrica (Phong)...")

    for indice, face in enumerate(faces_peao):
        v0_mundo, v1_mundo, v2_mundo = vertices_mundo[face[0]], vertices_mundo[face[1]], vertices_mundo[face[2]]
        
        # Culling
        normal_face = np.cross(v1_mundo - v0_mundo, v2_mundo - v0_mundo)
        if np.dot(normal_face, posicao_camera - v0_mundo) <= 0:
            continue
            
        # Calcula a iluminação EXATA (Phong) para as 3 pontas do triângulo
        cores = []
        for v_idx in face:
            n_homogeneo = np.array([norm_peao[v_idx][0], norm_peao[v_idx][1], norm_peao[v_idx][2], 0.0])
            n_mundo = np.dot(matriz_mundo_peao, n_homogeneo)[:3]
            
            # Ajustei os coeficientes (ka, kd, ks) para clarear a peça
            cor = illumination.calcular_iluminacao_phong(
                vertice_mundo=vertices_mundo[v_idx], normal=n_mundo,
                posicao_luz=posicao_luz, posicao_camera=posicao_camera,
                cor_objeto=[0.8, 0.1, 0.1], coeficientes=(0.4, 0.6, 0.8), brilho=64
            )
            cores.append(cor)
            
        pA, pB, pC = vertices_tela[face[0]], vertices_tela[face[1]], vertices_tela[face[2]]
        zA, zB, zC = profundidades_z[face[0]], profundidades_z[face[1]], profundidades_z[face[2]]
        
        triangulo_2d = [pA, pB, pC]
        pixels_internos = rasterizer.scan_line_par_impar(triangulo_2d)
        
        for (x, y) in pixels_internos:
            # INTERPOLAÇÃO BARICÊNTRICA
            alpha, beta, gamma = rasterizer.coordenadas_baricentricas(x, y, pA[0], pA[1], pB[0], pB[1], pC[0], pC[1])
            
            # Se o pixel pertence ao triângulo
            if alpha >= 0 and beta >= 0 and gamma >= 0:
                z_pixel = alpha * zA + beta * zB + gamma * zC
                
                # Z-BUFFER TEST
                if z_pixel < z_buffer[y, x]:
                    z_buffer[y, x] = z_pixel
                    
                    # Interpola a cor misturando os 3 vértices
                    cor_pixel = alpha * cores[0] + beta * cores[1] + gamma * cores[2]
                    imagem[y, x] = (np.clip(cor_pixel, 0.0, 1.0) * 255).astype(np.uint8)

    print("Sucesso!")
    
    # ---------------------------------------------------------
    # FASE 6: Apresentação Gráfica
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.imshow(imagem)
    plt.title("Peão Rasterizado (Z-Buffer + Interpolação Baricêntrica)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()