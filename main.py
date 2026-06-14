import numpy as np
import matplotlib.pyplot as plt
import modeling
import illumination
import math_3d
import rasterizer

# ==============================================================================
# CONFIGURAÇÃO DOS INPUTS DA CENA (Mapeado conforme o novo padrão do trio)
# Cores fortes via Hexadecimal convertidas para RGB float em tempo de execução
# ==============================================================================
CENA = [
    # (função_de_campo, posição_x, posição_z, cor_hex, nome_label, escala, limites_grade)
    (modeling.gerar_campo_peao, -7.5, -4.5, "#E63946", "Peão", 1.0, None),
    (modeling.gerar_campo_dama, -4.5, 1.5, "#F4A261", "Dama", 1.0, (-4, 4)),
    (modeling.gerar_campo_torre, -1.5, -4.5, "#2A9D8F", "Torre", 1.0, None),
    (modeling.gerar_campo_bispo, 1.5, 1.5, "#A8DADC", "Bispo", 1.0, None),
    (modeling.gerar_campo_rainha, 4.5, -4.5, "#C77DFF", "Rainha", 1.0, None),
    (modeling.gerar_campo_rei, 7.5, 1.5, "#FFD166", "Rei", 1.0, None),
]

def hex_to_rgb(hex_str):
    """Converte strings hexadecimais (#RRGGBB) para tuplas RGB float [0.0, 1.0]"""
    hex_str = hex_str.lstrip('#')
    return [int(hex_str[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def construir_matriz_mundo(px, pz, escala=1.0, rotacao_y=0.0, py_offset=-2.0):
    """Matriz homogênea 4x4 do Espaço do Mundo (Mapeada com py_offset=-2.0 do trio)"""
    m_e = math_3d.escala(escala, escala, escala)
    m_r = math_3d.rotacao_y(rotacao_y)
    m_t = math_3d.translacao(px, py_offset, pz)
    return m_t @ m_r @ m_e


# ==============================================================================
# RASTERIZADORES ESTRUTURAIS NA UNHA E MAPEAMENTO DE TICKS 2D
# ==============================================================================
def calcular_posicao_tela_rotulo(p_mundo, view_matrix, proj_matrix, res):
    """Converte um ponto 3D do mundo em coordenadas 2D de pixel para os rótulos numéricos"""
    v_cam = view_matrix @ np.array([p_mundo[0], p_mundo[1], p_mundo[2], 1.0])
    v_ndc = math_3d.divisao_perspectiva(proj_matrix @ v_cam)
    if abs(v_ndc[0]) > 1.0 or abs(v_ndc[1]) > 1.0:
        return None
    x_tela = (v_ndc[0] + 1.0) * 0.5 * res
    y_tela = (1.0 - v_ndc[1]) * 0.5 * res
    return (x_tela, y_tela)


def gerar_linhas_camera_blender(eye, at, tamanho=3.5):
    """Gera os segmentos de reta 3D que desenham o corpo físico da câmera (Pirâmide)"""
    # Normalização purista via NumPy para mitigar ausência de atributo no math_3d
    vetor_n = eye - at
    n = vetor_n / np.linalg.norm(vetor_n)

    up_ficticio = np.array([0.0, 1.0, 0.0])
    vetor_u = np.cross(up_ficticio, n)
    u = vetor_u / np.linalg.norm(vetor_u)

    v = np.cross(n, u)

    centro_base = eye - n * tamanho
    c1 = centro_base + u * (tamanho * 0.7) + v * (tamanho * 0.5)
    c2 = centro_base - u * (tamanho * 0.7) + v * (tamanho * 0.5)
    c3 = centro_base - u * (tamanho * 0.7) - v * (tamanho * 0.5)
    c4 = centro_base + u * (tamanho * 0.7) - v * (tamanho * 0.5)
    p_topo = centro_base + v * (tamanho * 0.8)

    arestas = [
        (eye, c1), (eye, c2), (eye, c3), (eye, c4),
        (c1, c2), (c2, c3), (c3, c4), (c4, c1),
        (centro_base, p_topo), (p_topo, c1)
    ]
    return arestas


def desenhar_reta_3d_no_zbuffer(p1_mundo, p2_mundo, cor, imagem, z_buffer, view_matrix, proj_matrix, res):
    """Projeta e rasteriza uma linha 3D no Z-Buffer usando o algoritmo de Bresenham"""
    v1_cam = view_matrix @ np.array([p1_mundo[0], p1_mundo[1], p1_mundo[2], 1.0])
    v1_ndc = math_3d.divisao_perspectiva(proj_matrix @ v1_cam)

    v2_cam = view_matrix @ np.array([p2_mundo[0], p2_mundo[1], p2_mundo[2], 1.0])
    v2_ndc = math_3d.divisao_perspectiva(proj_matrix @ v2_cam)

    if abs(v1_ndc[0]) > 1.8 or abs(v1_ndc[1]) > 1.8 or abs(v2_ndc[0]) > 1.8 or abs(v2_ndc[1]) > 1.8:
        return

    x1 = int((v1_ndc[0] + 1.0) * 0.5 * res)
    y1 = int((1.0 - v1_ndc[1]) * 0.5 * res)
    x2 = int((v2_ndc[0] + 1.0) * 0.5 * res)
    y2 = int((1.0 - v2_ndc[1]) * 0.5 * res)

    pontos_linha = rasterizer.rasterizar_bresenham(x1, y1, x2, y2)
    num_pontos = len(pontos_linha)
    if num_pontos == 0: return

    for idx, (px, py) in enumerate(pontos_linha):
        if 0 <= px < res and 0 <= py < res:
            t = idx / num_pontos
            z_interpolado = (1 - t) * v1_ndc[2] + t * v2_ndc[2]
            if z_interpolado < z_buffer[py, px] + 0.005:
                z_buffer[py, px] = z_interpolado
                imagem[py, px] = np.array(cor, dtype=np.uint8)


def desenhar_caixa_proporcao_mundo(imagem, z_buffer, view_matrix, proj_matrix, res):
    """Gera o grid tridimensional completo limitando o universo das peças"""
    cor_caixa = [205, 215, 228]
    cor_grade = [234, 240, 246]

    lim_x_min, lim_x_max = -15.0, 15.0  # Expandido para cobrir o Peão e Rei novos em +-12.5
    lim_z_min, lim_z_max = -10.0, 10.0
    y_chao = -2.0
    y_teto = 7.0

    # 1. Linhas de grade no chão divididas de 5 em 5 unidades
    for gx in np.arange(lim_x_min, lim_x_max + 1.0, 5.0):
        desenhar_reta_3d_no_zbuffer([gx, y_chao, lim_z_min], [gx, y_chao, lim_z_max], cor_grade, imagem, z_buffer,
                                    view_matrix, proj_matrix, res)
    for gz in np.arange(lim_z_min, lim_z_max + 1.0, 5.0):
        desenhar_reta_3d_no_zbuffer([lim_x_min, y_chao, gz], [lim_x_max, y_chao, gz], cor_grade, imagem, z_buffer,
                                    view_matrix, proj_matrix, res)

    # 2. Estrutura de Gaiola Cúbica Completa (Proporção do Mundo)
    arestas_caixa = [
        ([lim_x_min, y_chao, lim_z_min], [lim_x_max, y_chao, lim_z_min]),
        ([lim_x_min, y_chao, lim_z_max], [lim_x_max, y_chao, lim_z_max]),
        ([lim_x_min, y_chao, lim_z_min], [lim_x_min, y_chao, lim_z_max]),
        ([lim_x_max, y_chao, lim_z_min], [lim_x_max, y_chao, lim_z_max]),
        ([lim_x_min, y_chao, lim_z_min], [lim_x_min, y_teto, lim_z_min]),
        ([lim_x_max, y_chao, lim_z_min], [lim_x_max, y_teto, lim_z_min]),
        ([lim_x_min, y_chao, lim_z_max], [lim_x_min, y_teto, lim_z_max]),
        ([lim_x_max, y_chao, lim_z_max], [lim_x_max, y_teto, lim_z_max]),
        ([lim_x_min, y_teto, lim_z_min], [lim_x_max, y_teto, lim_z_min]),
        ([lim_x_min, y_teto, lim_z_max], [lim_x_max, y_teto, lim_z_max]),
        ([lim_x_min, y_teto, lim_z_min], [lim_x_min, y_teto, lim_z_max]),
        ([lim_x_max, y_teto, lim_z_min], [lim_x_max, y_teto, lim_z_max]),
    ]
    for p1, p2 in arestas_caixa:
        desenhar_reta_3d_no_zbuffer(p1, p2, cor_caixa, imagem, z_buffer, view_matrix, proj_matrix, res)


# ==============================================================================
# PIPELINE DE RENDERIZAÇÃO SÓLIDA (COM PHONG E SCANLINE)
# ==============================================================================
def renderizar_cena_completa(resolucao_tela, view_matrix, proj_matrix, cache_malhas, posicao_camera, posicao_luz,
                             modo_blender=False):
    """Pipeline aplicando Culling, Phong, Z-Buffer e Interpolação Baricêntrica."""
    imagem = np.full((resolucao_tela, resolucao_tela, 3), 245, dtype=np.uint8)
    z_buffer = np.full((resolucao_tela, resolucao_tela), np.inf)

    desenhar_caixa_proporcao_mundo(imagem, z_buffer, view_matrix, proj_matrix, resolucao_tela)

    for func_campo, px, pz, cor_hex, label, escala, limites in CENA:
        nome = func_campo.__name__
        cache_key = nome + str(limites)
        verts, faces, normais = cache_malhas[cache_key]
        matriz_mundo = construir_matriz_mundo(px, pz, escala=escala)

        vertices_tela, profundidades_z, vertices_mundo, normais_mundo = [], [], [], []
        cor_base_float = hex_to_rgb(cor_hex)

        # 1. Transformação de Vértices e Normais
        for i in range(len(verts)):
            v_mundo = matriz_mundo @ np.array([verts[i][0], verts[i][1], verts[i][2], 1.0])
            vertices_mundo.append(v_mundo[:3])

            # CORREÇÃO: Multiplicamos as normais por -1 para que apontem para FORA da peça
            n_mundo = (matriz_mundo @ np.array([-normais[i][0], -normais[i][1], -normais[i][2], 0.0]))[:3]
            normais_mundo.append(n_mundo)

            v_ndc = math_3d.divisao_perspectiva(proj_matrix @ (view_matrix @ v_mundo))
            profundidades_z.append(v_ndc[2])
            vertices_tela.append(((v_ndc[0] + 1.0) * 0.5 * resolucao_tela, (1.0 - v_ndc[1]) * 0.5 * resolucao_tela))

        # 2. Processamento das Faces (Rasterização Sólida)
        for face in faces:
            v0_m, v1_m, v2_m = vertices_mundo[face[0]], vertices_mundo[face[1]], vertices_mundo[face[2]]

            # BACK-FACE CULLING
            normal_face = np.cross(v1_m - v0_m, v2_m - v0_m)
            if np.dot(normal_face, posicao_camera - v0_m) <= 0:
                continue

            # ILUMINAÇÃO PHONG nos 3 vértices
            cores_vertices = []
            for v_idx in face:
                cor = illumination.calcular_iluminacao_phong(
                    vertice_mundo=vertices_mundo[v_idx], normal=normais_mundo[v_idx],
                    posicao_luz=posicao_luz, posicao_camera=posicao_camera,
                    cor_objeto=cor_base_float,
                    # CORREÇÃO: Aumentamos a luz Ambiente (ka) para 0.5 e a Difusa (kd) para 0.8
                    coeficientes=(0.5, 0.8, 0.4), brilho=32
                )
                cores_vertices.append(cor)

            pA, pB, pC = vertices_tela[face[0]], vertices_tela[face[1]], vertices_tela[face[2]]
            zA, zB, zC = profundidades_z[face[0]], profundidades_z[face[1]], profundidades_z[face[2]]

            triangulo_2d = [pA, pB, pC]

            # SCAN LINE
            pixels_internos = rasterizer.scan_line_par_impar(triangulo_2d, largura_tela=resolucao_tela,
                                                             altura_tela=resolucao_tela)

            for (x, y) in pixels_internos:
                # INTERPOLAÇÃO BARICÊNTRICA
                alpha, beta, gamma = rasterizer.coordenadas_baricentricas(x, y, pA[0], pA[1], pB[0], pB[1], pC[0],
                                                                          pC[1])

                if alpha >= 0 and beta >= 0 and gamma >= 0:
                    z_pixel = alpha * zA + beta * zB + gamma * zC

                    if z_pixel < z_buffer[y, x]:
                        z_buffer[y, x] = z_pixel
                        cor_pixel = alpha * cores_vertices[0] + beta * cores_vertices[1] + gamma * cores_vertices[2]
                        imagem[y, x] = (np.clip(cor_pixel, 0.0, 1.0) * 255).astype(np.uint8)

    if modo_blender:
        r_origem = 0.4
        arestas_losango = [
            ([r_origem, 0, 0], [0, r_origem, 0]), ([0, r_origem, 0], [-r_origem, 0, 0]),
            ([-r_origem, 0, 0], [0, -r_origem, 0]), ([0, -r_origem, 0], [r_origem, 0, 0]),
            ([r_origem, 0, 0], [0, 0, r_origem]), ([0, 0, r_origem], [-r_origem, 0, 0]),
            ([-r_origem, 0, 0], [0, 0, -r_origem]), ([0, 0, -r_origem], [r_origem, 0, 0]),
            ([0, r_origem, 0], [0, 0, r_origem]), ([0, 0, r_origem], [0, -r_origem, 0]),
            ([0, -r_origem, 0], [0, 0, -r_origem]), ([0, 0, -r_origem], [0, r_origem, 0])
        ]
        for p1, p2 in arestas_losango:
            desenhar_reta_3d_no_zbuffer(np.array(p1) + [0, -1.9, 0], np.array(p2) + [0, -1.9, 0], [255, 110, 30],
                                        imagem, z_buffer, view_matrix, proj_matrix, resolucao_tela)

        posicao_camera_ficticia = np.array([0.0, 8.0, 25.0])
        linhas_cam = gerar_linhas_camera_blender(posicao_camera_ficticia, at=[0.0, 0.0, 0.0], tamanho=4.0)
        for p1, p2 in linhas_cam:
            desenhar_reta_3d_no_zbuffer(p1, p2, [255, 140, 0], imagem, z_buffer, view_matrix, proj_matrix,
                                        resolucao_tela)

    return imagem


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    print("=== PIPELINE ADAPTADO: NOVAS MALHAS PRIMITIVAS + DUPLO VIEWPORT ===")

    cache_malhas = {}
    total = len(CENA)
    for i, (func_campo, px, pz, cor_hex, label, escala, limites) in enumerate(CENA):
        nome = func_campo.__name__
        cache_key = nome + str(limites)
        print(f"  Extraindo Malha [{i + 1}/{total}] -> {label}")
        if cache_key not in cache_malhas:
            kwargs = {"limites": limites, "resolucao": 35} if limites is not None else {"resolucao": 35}
            verts, faces, normals = modeling.extrair_malha(func_campo, **kwargs)
            cache_malhas[cache_key] = (verts, faces, normals)

    posicao_camera_real = np.array([7.5, 10.0, 5.0])  #[0.0, 8.0, 25.0]
    posicao_luz_global = np.array([5.0, 20.0, 15.0])  # Luz adicionada aqui

    view_real = math_3d.look_at(eye=posicao_camera_real, at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    proj_real = math_3d.projecao_perspectiva(fov_graus=58, aspecto=1.0, z_near=0.1, z_far=100)

    RESOLUCOES = [200, 400, 800]
    import matplotlib.pyplot as plt
    plt.figure("Tela 1: Comparativo de Resoluções (Novas Peças Primitivas)", figsize=(18, 6))

    for idx, res in enumerate(RESOLUCOES):
        print(f"[Fase Raster] Processando Visão Interna em {res}x{res}...")
        # Note a adição de posicao_camera e posicao_luz_global nos argumentos abaixo
        img_res = renderizar_cena_completa(res, view_real, proj_real, cache_malhas, posicao_camera_real,
                                           posicao_luz_global, modo_blender=False)

        plt.subplot(1, 3, idx + 1)
        plt.imshow(img_res)
        plt.title(f"Resolução: {res} x {res}", fontsize=11, fontweight='bold', color='#2c3e50')
        plt.axis('off')

        for val in [-15, -10, -5, 0, 5, 10, 15]:
            pos_x_2d = calcular_posicao_tela_rotulo([val, -2.0, 10.0], view_real, proj_real, res)
            if pos_x_2d:
                plt.text(pos_x_2d[0], pos_x_2d[1] + (res * 0.04), str(val), color='#7f8c8d',
                         fontsize=max(6, int(res * 0.02)), ha='center', va='top', weight='bold')
        for val in [-10, -5, 0, 5, 10]:
            pos_z_2d = calcular_posicao_tela_rotulo([-15.0, -2.0, val], view_real, proj_real, res)
            if pos_z_2d:
                plt.text(pos_z_2d[0] - (res * 0.04), pos_z_2d[1], str(val), color='#7f8c8d',
                         fontsize=max(6, int(res * 0.02)), ha='right', va='center', weight='bold')

    plt.tight_layout()
    plt.show(block=False)

    olho_desenvolvimento = np.array([26.0, 18.0, 28.0])
    view_desenv = math_3d.look_at(eye=olho_desenvolvimento, at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    proj_desenv = math_3d.projecao_perspectiva(fov_graus=50, aspecto=1.0, z_near=0.1, z_far=100)

    print("\n[Fase Raster] Renderizando Viewport Externa (Estilo Blender) em 800x800...")
    render_blender = renderizar_cena_completa(800, view_desenv, proj_desenv, cache_malhas, olho_desenvolvimento,
                                              posicao_luz_global, modo_blender=True)

    plt.figure("Tela 2: Viewport de Trabalho (Estilo Blender 3D)", figsize=(8, 8))
    plt.imshow(render_blender)
    plt.title("Visualização Global (Diferenciação Mundo vs Câmera)", fontsize=12, fontweight='bold', color='#2c3e50')
    plt.axis('off')

    for val in [-15, -10, -5, 0, 5, 10, 15]:
        pos_x_2d = calcular_posicao_tela_rotulo([val, -2.0, 10.0], view_desenv, proj_desenv, 800)
        if pos_x_2d:
            plt.text(pos_x_2d[0], pos_x_2d[1] + 32, str(val), color='#7f8c8d', fontsize=11, ha='center', va='top',
                     weight='bold')
    for val in [-10, -5, 0, 5, 10]:
        pos_z_2d = calcular_posicao_tela_rotulo([-15.0, -2.0, val], view_desenv, proj_desenv, 800)
        if pos_z_2d:
            plt.text(pos_z_2d[0] - 32, pos_z_2d[1], str(val), color='#7f8c8d', fontsize=11, ha='right', va='center',
                     weight='bold')

    print("Sucesso!")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()