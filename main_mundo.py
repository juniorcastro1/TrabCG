import numpy as np
import matplotlib.pyplot as plt
import modeling
import math_3d
import illumination
import rasterizer

# ==============================================================================
# CONFIGURAÇÃO DOS INPUTS DA CENA (Questão 2, 2.a e 4)
# Todos os sólidos posicionados sem colisão e contidos no limite regulamentar de 10
# ==============================================================================
CENA = [
    # (função_de_campo, posição_x, posição_z, cor_objeto_rgb, nome_label, escala)
    (modeling.gerar_campo_peao_elaborado, -7.5, 1.5, [0.9, 0.1, 0.1], "Peão", 0.65),
    (modeling.gerar_campo_dama, -4.5, -1.5, [0.9, 0.5, 0.1], "Dama", 0.65),
    (modeling.gerar_campo_torre_elaborada, -1.5, 2.0, [0.1, 0.6, 0.5], "Torre", 0.65),
    (modeling.gerar_campo_bispo, 1.5, -1.0, [0.1, 0.5, 0.8], "Bispo", 0.65),
    (modeling.gerar_campo_rainha, 4.5, 1.0, [0.6, 0.2, 0.8], "Rainha", 0.55),
    (modeling.gerar_campo_rei, 7.5, -2.0, [0.9, 0.7, 0.1], "Rei", 0.45),
]


def construir_matriz_mundo(px, pz, escala=1.0, rotacao_y=0.0, py_offset=-1.5):
    """Matriz homogênea 4x4 do Espaço do Mundo"""
    m_e = math_3d.escala(escala, escala, escala)
    m_r = math_3d.rotacao_y(rotacao_y)
    m_t = math_3d.translacao(px, py_offset, pz)
    return m_t @ m_r @ m_e


# ==============================================================================
# GERADORES DE MALHA ADICIONAIS NA UNHA (Para emular o ambiente do Blender)
# ==============================================================================
def gerar_linhas_camera_blender(eye, at, tamanho=3.0):
    """Gera os segmentos de reta 3D que desenham o corpo físico da câmera"""
    n = math_3d.normalizar(eye - at)
    u = math_3d.normalizar(np.cross([0.0, 1.0, 0.0], n))
    v = np.cross(n, u)

    centro_base = eye - n * tamanho
    c1 = centro_base + u * (tamanho * 0.7) + v * (tamanho * 0.5)
    c2 = centro_base - u * (tamanho * 0.7) + v * (tamanho * 0.5)
    c3 = centro_base - u * (tamanho * 0.7) - v * (tamanho * 0.5)
    c4 = centro_base + u * (tamanho * 0.7) - v * (tamanho * 0.5)
    p_topo = centro_base + v * (tamanho * 0.8)

    # Segmentos de reta ligando os pontos para formar o aramado da pirâmide
    arestas = [
        (eye, c1), (eye, c2), (eye, c3), (eye, c4),
        (c1, c2), (c2, c3), (c3, c4), (c4, c1),
        (centro_base, p_topo), (p_topo, c1)
    ]
    return arestas


def desenhar_reta_3d_no_zbuffer(p1_mundo, p2_mundo, cor, imagem, z_buffer, view_matrix, proj_matrix, res):
    """Projeta e rasteriza uma linha 3D na unha usando o algoritmo de Bresenham"""
    # Transforma p1
    v1_cam = view_matrix @ np.array([p1_mundo[0], p1_mundo[1], p1_mundo[2], 1.0])
    v1_ndc = math_3d.divisao_perspectiva(proj_matrix @ v1_cam)
    # Transforma p2
    v2_cam = view_matrix @ np.array([p2_mundo[0], p2_mundo[1], p2_mundo[2], 1.0])
    v2_ndc = math_3d.divisao_perspectiva(proj_matrix @ v2_cam)

    # Clip manual simples de segurança para eixos infinitos
    if abs(v1_ndc[0]) > 2 or abs(v1_ndc[1]) > 2 or abs(v2_ndc[0]) > 2 or abs(v2_ndc[1]) > 2:
        return

    # Mapeamento de Viewport
    x1 = int((v1_ndc[0] + 1.0) * 0.5 * res)
    y1 = int((1.0 - v1_ndc[1]) * 0.5 * res)
    x2 = int((v2_ndc[0] + 1.0) * 0.5 * res)
    y2 = int((1.0 - v2_ndc[1]) * 0.5 * res)

    # Roda o Bresenham do seu próprio arquivo rasterizer.py!
    pontos_linha = rasterizer.rasterizar_bresenham(x1, y1, x2, y2)

    # Pinta os pixels da linha interpolando o Z para respeitar a oclusão das peças
    num_pontos = len(pontos_linha)
    if num_pontos == 0: return

    for idx, (px, py) in enumerate(pontos_linha):
        if 0 <= px < res and 0 <= py < res:
            t = idx / num_pontos
            z_interpolado = (1 - t) * v1_ndc[2] + t * v2_ndc[2]

            # Teste de profundidade para que a linha passe por trás/frente das peças corretamente
            if z_interpolado < z_buffer[py, px]:
                z_buffer[py, px] = z_interpolado
                imagem[py, px] = np.array(cor, dtype=np.uint8)


# ==============================================================================
# PIPELINE DE RENDERIZAÇÃO PRINCIPAL
# ==============================================================================
def renderizar_cena_completa(resolucao_tela, view_matrix, proj_matrix, posicao_camera, posicao_luz, cache_malhas,
                             modo_blender=False):
    imagem = np.zeros((resolucao_tela, resolucao_tela, 3), dtype=np.uint8)
    z_buffer = np.full((resolucao_tela, resolucao_tela), np.inf)

    # 1. RASTERIZAÇÃO DOS SÓLIDOS (Triângulo por Triângulo)
    for func_campo, px, pz, cor_objeto, label, escala in CENA:
        nome = func_campo.__name__
        verts, faces, normals = cache_malhas[nome]
        matriz_mundo = construir_matriz_mundo(px, pz, escala=escala)

        vertices_mundo, vertices_tela, profundidades_z, normais_mundo = [], [], [], []

        for i in range(len(verts)):
            v_mundo = matriz_mundo @ np.array([verts[i][0], verts[i][1], verts[i][2], 1.0])
            vertices_mundo.append(v_mundo[:3])
            normais_mundo.append((matriz_mundo @ np.array([normals[i][0], normals[i][1], normals[i][2], 0.0]))[:3])

            v_ndc = math_3d.divisao_perspectiva(proj_matrix @ (view_matrix @ v_mundo))
            profundidades_z.append(v_ndc[2])
            vertices_tela.append(((v_ndc[0] + 1.0) * 0.5 * resolucao_tela, (1.0 - v_ndc[1]) * 0.5 * resolucao_tela))

        for face in faces:
            v0_m, v1_m, v2_m = vertices_mundo[face[0]], vertices_mundo[face[1]], vertices_mundo[face[2]]
            normal_face = np.cross(v1_m - v0_m, v2_m - v0_m)

            # Back-face Culling analítico na unha
            if np.dot(normal_face, posicao_camera - v0_m) <= 0:
                continue

            cores_vertices = []
            for v_idx in face:
                cores_vertices.append(illumination.calcular_iluminacao_phong(
                    vertices_mundo[v_idx], normais_mundo[v_idx], posicao_luz, posicao_camera, cor_objeto,
                    (0.3, 0.7, 0.5), 32
                ))

            pA, pB, pC = vertices_tela[face[0]], vertices_tela[face[1]], vertices_tela[face[2]]
            zA, zB, zC = profundidades_z[face[0]], profundidades_z[face[1]], profundidades_z[face[2]]

            pixels_internos = rasterizer.scan_line_par_impar([pA, pB, pC], resolucao_tela, resolucao_tela)

            for (x, y) in pixels_internos:
                alpha, beta, gamma = rasterizer.coordenadas_baricentricas(x, y, pA[0], pA[1], pB[0], pB[1], pC[0],
                                                                          pC[1])
                if alpha >= 0 and beta >= 0 and gamma >= 0:
                    z_pixel = alpha * zA + beta * zB + gamma * zC
                    if z_pixel < z_buffer[y, x]:
                        z_buffer[y, x] = z_pixel
                        cor_final = alpha * cores_vertices[0] + beta * cores_vertices[1] + gamma * cores_vertices[2]
                        imagem[y, x] = (np.clip(cor_final, 0.0, 1.0) * 255).astype(np.uint8)

    # 2. SE ESTIVERMOS NO MODO VIEWPORT EXTERNA (EMULAÇÃO DE BLENDER NA UNHA)
    if modo_blender:
        # Desenha o Eixo X (Linha Vermelha no chão)
        desenhar_reta_3d_no_zbuffer([-15, -1.5, 0], [15, -1.5, 0], [255, 50, 50], imagem, z_buffer, view_matrix,
                                    proj_matrix, resolucao_tela)
        # Desenha o Eixo Z (Linha Verde no chão)
        desenhar_reta_3d_no_zbuffer([0, -1.5, -15], [0, -1.5, 15], [50, 255, 50], imagem, z_buffer, view_matrix,
                                    proj_matrix, resolucao_tela)

        # Desenha um Losango Tridimensional na Origem (0,0,0) - Marcador de Diamante Laranja Corrigido
        r_origem = 0.3
        arestas_losango = [
            ([r_origem, 0, 0], [0, r_origem, 0]), ([0, r_origem, 0], [-r_origem, 0, 0]),
            ([-r_origem, 0, 0], [0, -r_origem, 0]), ([0, -r_origem, 0], [r_origem, 0, 0]),
            ([r_origem, 0, 0], [0, 0, r_origem]), ([0, 0, r_origem], [-r_origem, 0, 0]),
            ([-r_origem, 0, 0], [0, 0, -r_origem]), ([0, 0, -r_origem], [r_origem, 0, 0]),
            ([0, r_origem, 0], [0, 0, r_origem]), ([0, 0, r_origem], [0, -r_origem, 0]),
            ([0, -r_origem, 0], [0, 0, -r_origem]), ([0, 0, -r_origem], [0, r_origem, 0])
        ]
        for p1, p2 in arestas_losango:
            # Desloca ligeiramente para cima do grid do chão
            desenhar_reta_3d_no_zbuffer(np.array(p1) + [0, -1.4, 0], np.array(p2) + [0, -1.4, 0], [255, 120, 50],
                                        imagem, z_buffer, view_matrix, proj_matrix, resolucao_tela)

        # Desenha a Pirâmide da Câmera Laranja Fixa Física olhando a cena (A posição 'eye' original do render)
        posicao_camera_ficticia = np.array([12.0, 8.0, 15.0]) #12.0, 8.0, 150.0
        linhas_cam = gerar_linhas_camera_blender(posicao_camera_ficticia, at=[0.0, 0.0, 0.0], tamanho=2.5)
        for p1, p2 in linhas_cam:
            desenhar_reta_3d_no_zbuffer(p1, p2, [255, 160, 20], imagem, z_buffer, view_matrix, proj_matrix,
                                        resolucao_tela)

    return imagem


def main():
    print("=== RENDERS TIPO PLOTLY GERADOS TOTALMENTE NA UNHA ===")

    # Parâmetros Universais da Lâmpada e Malhas
    posicao_luz = np.array([5.0, 15.0, 10.0])
    cache_malhas = {f.__name__: modeling.extrair_malha(f, resolucao=55) for f, _, _, _, _, _ in CENA}

    # --------------------------------------------------------------------------
    # QUADRO 1: A VIEWPORT GLOBAL DO BLENDER (Olhando de fora com Câmera e Eixos)
    # --------------------------------------------------------------------------
    olho_desenvolvimento = np.array([22.0, 15.0, 25.0])
    lv_desenv = np.array([0.0, 0.0, 0.0]) - olho_desenvolvimento
    up_desenv = math_3d.normalizar(np.cross(np.cross(lv_desenv, [0.0, 1.0, 0.0]), lv_desenv))

    view_desenv = math_3d.look_at(eye=olho_desenvolvimento, at=[0.0, 0.0, 0.0], up=up_desenv)
    proj_desenv = math_3d.projecao_perspectiva(fov_graus=45, aspecto=1.0, z_near=0.1, z_far=100)

    print("\n[Tela 1/2] Renderizando Viewport Externa (Estilo Blender)...")
    render_blender = renderizar_cena_completa(
        resolucao_tela=800, view_matrix=view_desenv, proj_matrix=proj_desenv,
        posicao_camera=olho_desenvolvimento, posicao_luz=posicao_luz, cache_malhas=cache_malhas, modo_blender=True
    )

    # --------------------------------------------------------------------------
    # QUADRO 2: O RENDER REAL DA CÂMERA (O que a pirâmide laranja está capturando)
    # --------------------------------------------------------------------------
    posicao_camera_real = np.array([12.0, 8.0, 15.0])
    lv_real = np.array([0.0, 0.0, 0.0]) - posicao_camera_real
    up_real = math_3d.normalizar(np.cross(np.cross(lv_real, [0.0, 1.0, 0.0]), lv_real))

    view_real = math_3d.look_at(eye=posicao_camera_real, at=[0.0, 0.0, 0.0], up=up_real)
    proj_real = math_3d.projecao_perspectiva(fov_graus=55, aspecto=1.0, z_near=0.1, z_far=100)

    print("[Tela 2/2] Renderizando Visão Interna (O que a Câmera Enxerga)...")
    render_camera_interna = renderizar_cena_completa(
        resolucao_tela=800, view_matrix=view_real, proj_matrix=proj_real,
        posicao_camera=posicao_camera_real, posicao_luz=posicao_luz, cache_malhas=cache_malhas, modo_blender=False
    )

    # EXIBIÇÃO NO MATPLOTLIB
    plt.figure(figsize=(14, 7))

    # Lado Esquerdo: A Viewport de Trabalho
    plt.subplot(1, 2, 1)
    plt.imshow(render_blender)
    plt.title("Viewport 3D Global (Na Unha — Eixos + Câmera Física Laranja Visível)")
    plt.axis('off')

    # Lado Direito: A captura de dentro da câmera
    plt.subplot(1, 2, 2)
    plt.imshow(render_camera_interna)
    plt.title("Visão Interna Capturada Pela Lente da Câmera (Perspectiva 2D)")
    plt.axis('off')

    print("\nSucesso! Exibindo renders finais...")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()