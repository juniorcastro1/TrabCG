import numpy as np
import matplotlib.pyplot as plt
import modeling
import math_3d
import illumination
import rasterizer

# ==============================================================================
# CONFIGURAÇÃO DOS INPUTS DA CENA (Questão 2, 2.a e 4)
# Cada objeto possui sua cor RGB forte e saturada para cumprir o Item 4 do PDF
# ==============================================================================
CENA = [
    # (função_de_campo, posição_x, posição_z, cor_objeto_rgb, nome_label, escala)
    (modeling.gerar_campo_peao_elaborado, -7.5, 1.5, [1.0, 0.0, 0.0], "Peão", 0.65),  # Vermelho Puro
    (modeling.gerar_campo_dama, -4.5, -1.5, [1.0, 0.4, 0.0], "Dama", 0.65),  # Laranja Forte
    (modeling.gerar_campo_torre_elaborada, -1.5, 2.0, [0.0, 1.0, 0.5], "Torre", 0.65),  # Verde-Água Vivo
    (modeling.gerar_campo_bispo, 1.5, -1.0, [0.0, 0.0, 1.0], "Bispo", 0.65),  # Azul Puro
    (modeling.gerar_campo_rainha, 4.5, 1.0, [0.7, 0.0, 1.0], "Rainha", 0.55),  # Roxo Intenso
    (modeling.gerar_campo_rei, 7.5, -2.0, [1.0, 0.9, 0.0], "Rei", 0.45),  # Amarelo Vivo
]


def construir_matriz_mundo(px, pz, escala=1.0, rotacao_y=0.0, py_offset=-1.5):
    """Matriz homogênea 4x4 do Espaço do Mundo (Questão 2)"""
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


def gerar_linhas_camera_blender(eye, at, tamanho=3.0):
    """Gera os segmentos de reta 3D que desenham o corpo físico da câmera (Pirâmide)"""
    n = math_3d.normalizar(eye - at)
    u = math_3d.normalizar(np.cross([0.0, 1.0, 0.0], n))
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
    """Gera o grid tridimensional completo limitando o universo de 10 unidades"""
    cor_caixa = [205, 215, 228]
    cor_grade = [234, 240, 246]

    lim_min, lim_max = -10.0, 10.0
    y_chao = -1.5
    y_teto = 6.0

    # 1. Linhas internas do chão divididas de 5 em 5 unidades (Questão 2.a)
    for g in np.arange(lim_min, lim_max + 1.0, 5.0):
        desenhar_reta_3d_no_zbuffer([g, y_chao, lim_min], [g, y_chao, lim_max], cor_grade, imagem, z_buffer,
                                    view_matrix, proj_matrix, res)
        desenhar_reta_3d_no_zbuffer([lim_min, y_chao, g], [lim_max, y_chao, g], cor_grade, imagem, z_buffer,
                                    view_matrix, proj_matrix, res)

    # 2. Estrutura de Gaiola Cúbica Comple (Proporção do Mundo)
    arestas_caixa = [
        ([lim_min, y_chao, lim_min], [lim_max, y_chao, lim_min]),
        ([lim_min, y_chao, lim_max], [lim_max, y_chao, lim_max]),
        ([lim_min, y_chao, lim_min], [lim_min, y_chao, lim_max]),
        ([lim_max, y_chao, lim_min], [lim_max, y_chao, lim_max]),
        ([lim_min, y_chao, lim_min], [lim_min, y_teto, lim_min]),
        ([lim_max, y_chao, lim_min], [lim_max, y_teto, lim_min]),
        ([lim_min, y_chao, lim_max], [lim_min, y_teto, lim_max]),
        ([lim_max, y_chao, lim_max], [lim_max, y_teto, lim_max]),
        ([lim_min, y_teto, lim_min], [lim_max, y_teto, lim_min]),
        ([lim_min, y_teto, lim_max], [lim_max, y_teto, lim_max]),
        ([lim_min, y_teto, lim_min], [lim_min, y_teto, lim_max]),
        ([lim_max, y_teto, lim_min], [lim_max, y_teto, lim_max]),
    ]
    for p1, p2 in arestas_caixa:
        desenhar_reta_3d_no_zbuffer(p1, p2, cor_caixa, imagem, z_buffer, view_matrix, proj_matrix, res)


# ==============================================================================
# PIPELINE PRINCIPAL DE RENDERIZAÇÃO ESTILO PLOTLY POINT-MESH MULTI-COR
# ==============================================================================
def renderizar_cena_completa(resolucao_tela, view_matrix, proj_matrix, cache_malhas, modo_blender=False):
    """Pipeline analítico na unha simulando a estética por malhas de linhas independentes do Plotly."""
    imagem = np.full((resolucao_tela, resolucao_tela, 3), 245, dtype=np.uint8)
    z_buffer = np.full((resolucao_tela, resolucao_tela), np.inf)

    # Desenha o sistema de proporção de eixos e caixa tridimensional no fundo
    desenhar_caixa_proporcao_mundo(imagem, z_buffer, view_matrix, proj_matrix, resolucao_tela)

    # Loop de processamento das peças
    for func_campo, px, pz, cor_objeto, label, escala in CENA:
        nome = func_campo.__name__
        verts, faces, _ = cache_malhas[nome]
        matriz_mundo = construir_matriz_mundo(px, pz, escala=escala)

        vertices_tela, profundidades_z = [], []

        for i in range(len(verts)):
            v_mundo = matriz_mundo @ np.array([verts[i][0], verts[i][1], verts[i][2], 1.0])
            v_ndc = math_3d.divisao_perspectiva(proj_matrix @ (view_matrix @ v_mundo))
            profundidades_z.append(v_ndc[2])
            vertices_tela.append(((v_ndc[0] + 1.0) * 0.5 * resolucao_tela, (1.0 - v_ndc[1]) * 0.5 * resolucao_tela))

        cor_rgb_peca = (np.array(cor_objeto) * 255).astype(np.uint8)

        for face in faces:
            pA, pB, pC = vertices_tela[face[0]], vertices_tela[face[1]], vertices_tela[face[2]]
            zA, zB, zC = profundidades_z[face[0]], profundidades_z[face[1]], profundidades_z[face[2]]

            arestas = [(pA, pB, zA, zB), (pB, pC, zB, zC), (pC, pA, zC, zA)]
            for p1, p2, z1, z2 in arestas:
                pontos = rasterizer.rasterizar_bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
                num_p = len(pontos)
                if num_p == 0: continue

                for idx, (bx, by) in enumerate(pontos):
                    if 0 <= bx < resolucao_tela and 0 <= by < resolucao_tela:
                        t = idx / num_p
                        z_lin = (1 - t) * z1 + t * z2
                        if z_lin < z_buffer[by, bx]:
                            z_buffer[by, bx] = z_lin
                            imagem[by, bx] = cor_rgb_peca

    # Adiciona a pirâmide da câmera e marcador de origem caso esteja ativo
    if modo_blender:
        # Desenha Diamante Laranja na Origem (0,0,0) - Item 3.d do PDF
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
            desenhar_reta_3d_no_zbuffer(np.array(p1) + [0, -1.4, 0], np.array(p2) + [0, -1.4, 0], [255, 110, 30],
                                        imagem, z_buffer, view_matrix, proj_matrix, resolucao_tela)

        # Desenha a Pirâmide Física da Câmera Laranja no ponto real
        posicao_camera_ficticia = np.array([12.0, 8.0, 15.0])
        linhas_cam = gerar_linhas_camera_blender(posicao_camera_ficticia, at=[0.0, 0.0, 0.0], tamanho=2.5)
        for p1, p2 in linhas_cam:
            desenhar_reta_3d_no_zbuffer(p1, p2, [255, 140, 0], imagem, z_buffer, view_matrix, proj_matrix,
                                        resolucao_tela)

    return imagem


# ==============================================================================
# EXECUÇÃO PRINCIPAL DO DUPLO PIPELINE SIMULTÂNEO (Questão 6 e Obs1)
# ==============================================================================
def main():
    print("=== PIPELINE INTEGRADO: MULTI-RESOLUÇÃO + VIEWPORT COMPLETA ===")
    cache_malhas = {f.__name__: modeling.extrair_malha(f, resolucao=35) for f, _, _, _, _, _ in CENA}

    # Matrizes da Câmera Real (O que gera o enquadramento interno)
    posicao_camera_real = np.array([12.0, 8.0, 15.0])
    view_real = math_3d.look_at(eye=posicao_camera_real, at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    proj_real = math_3d.projecao_perspectiva(fov_graus=55, aspecto=1.0, z_near=0.1, z_far=100)

    # --------------------------------------------------------------------------
    # TELA 1: AS 3 RESOLUÇÕES SIMULTÂNEAS (Cores Fortes + Ticks)
    # --------------------------------------------------------------------------
    RESOLUCOES = [200, 400, 800]
    plt.figure("Tela 1: Comparativo de Resoluções (Visão Interna)", figsize=(18, 6))

    for idx, res in enumerate(RESOLUCOES):
        print(f"[Fase Raster] Processando Visão Interna em {res}x{res}...")
        img_res = renderizar_cena_completa(res, view_real, proj_real, cache_malhas, modo_blender=False)

        plt.subplot(1, 3, idx + 1)
        plt.imshow(img_res)
        plt.title(f"Resolução: {res} x {res}", fontsize=11, fontweight='bold', color='#2c3e50')
        plt.axis('off')

        # Desenha os Ticks numéricos de proporção na Tela 1
        for val in [-10, -5, 0, 5, 10]:
            pos_x_2d = calcular_posicao_tela_rotulo([val, -1.5, 10.0], view_real, proj_real, res)
            if pos_x_2d:
                plt.text(pos_x_2d[0], pos_x_2d[1] + (res * 0.04), str(val), color='#7f8c8d',
                         fontsize=max(6, int(res * 0.02)), ha='center', va='top', weight='bold')
            pos_z_2d = calcular_posicao_tela_rotulo([-10.0, -1.5, val], view_real, proj_real, res)
            if pos_z_2d:
                plt.text(pos_z_2d[0] - (res * 0.04), pos_z_2d[1], str(val), color='#7f8c8d',
                         fontsize=max(6, int(res * 0.02)), ha='right', va='center', weight='bold')

    plt.tight_layout()
    plt.show(block=False)  # Exibe a Tela 1 sem bloquear o terminal

    # --------------------------------------------------------------------------
    # TELA 2: VIEWPORT GLOBAL 3D (Visão Externa Inclinada Estilo Blender)
    # --------------------------------------------------------------------------
    olho_desenvolvimento = np.array([22.0, 15.0, 25.0])
    view_desenv = math_3d.look_at(eye=olho_desenvolvimento, at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    proj_desenv = math_3d.projecao_perspectiva(fov_graus=45, aspecto=1.0, z_near=0.1, z_far=100)

    print("\n[Fase Raster] Renderizando Viewport Externa (Blender 3D) em 800x800...")
    render_blender = renderizar_cena_completa(800, view_desenv, proj_desenv, cache_malhas, modo_blender=True)

    plt.figure("Tela 2: Viewport de Trabalho (Estilo Blender 3D)", figsize=(8, 8))
    plt.imshow(render_blender)
    plt.title("Visualização Global (Diferenciação Mundo vs Câmera)", fontsize=12, fontweight='bold', color='#2c3e50')
    plt.axis('off')

    # Rótulos de Ticks da Viewport Externa para fechar a proporção de fora
    for val in [-10, -5, 0, 5, 10]:
        pos_x_2d = calcular_posicao_tela_rotulo([val, -1.5, 10.0], view_desenv, proj_desenv, 800)
        if pos_x_2d:
            plt.text(pos_x_2d[0], pos_x_2d[1] + 32, str(val), color='#7f8c8d', fontsize=12, ha='center', va='top',
                     weight='bold')
        pos_z_2d = calcular_posicao_tela_rotulo([-10.0, -1.5, val], view_desenv, proj_desenv, 800)
        if pos_z_2d:
            plt.text(pos_z_2d[0] - 32, pos_z_2d[1], str(val), color='#7f8c8d', fontsize=12, ha='right', va='center',
                     weight='bold')

    print("Sucesso! Ambas as janelas carregadas simultaneamente.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()