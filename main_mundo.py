import numpy as np
import plotly.graph_objects as go
import modeling
import math_3d

# ==============================================================================
# DEFINIÇÃO DA CENA
# Cada entrada: (função_de_campo, posição_x, posição_z, cor_hex, nome_label, escala)
# ==============================================================================
# Uma peça de cada tipo, distribuídas em linha, cada uma com cor distinta
# (função_de_campo, posição_x, posição_z, cor_hex, nome_label, escala)
CENA = [
    (modeling.gerar_campo_peao, -12.5, 0.0, "#E63946", "Peão", 1.0),
    (modeling.gerar_campo_dama,    -7.5, 0.0, "#F4A261", "Dama",   1.0),
    (modeling.gerar_campo_torre, -2.5, 0.0, "#2A9D8F", "Torre",  1.0),
    (modeling.gerar_campo_bispo,    2.5, 0.0, "#A8DADC", "Bispo",  1.0),
    (modeling.gerar_campo_rainha,   7.5, 0.0, "#C77DFF", "Rainha", 1.0),
    (modeling.gerar_campo_rei,     12.5, 0.0, "#FFD166", "Rei",    1.0),
]


def construir_matriz_mundo(px, pz, escala=1.0, rotacao_y=0.0, py_offset=-2.0):
    """Translação + rotação Y + escala uniforme para posicionar a peça no mundo."""
    m_e = math_3d.escala(escala, escala, escala)
    m_r = math_3d.rotacao_y(rotacao_y)
    m_t = math_3d.translacao(px, py_offset, pz)
    return m_t @ m_r @ m_e


def aplicar_matriz_mundo(vertices, matriz_mundo):
    """Transforma um array (N,3) de vértices locais para o espaço mundo."""
    ones   = np.ones((len(vertices), 1))
    v_hom  = np.hstack([vertices, ones])          # (N, 4)
    v_mundo = (matriz_mundo @ v_hom.T).T          # (N, 4)
    return v_mundo[:, :3]


def main():
    print("Gerando cena 3D com Plotly...")

    traces = []

    # --- Cache de malhas ---
    cache_malhas = {}
    total = len(CENA)

    for i, (func_campo, px, pz, cor_hex, label, escala) in enumerate(CENA):
        nome = func_campo.__name__
        print(f"  [{i+1}/{total}] {label} em ({px:.0f}, {pz:.0f})")

        if nome not in cache_malhas:
            verts, faces, normals = modeling.extrair_malha(func_campo)
            cache_malhas[nome] = (verts, faces, normals)
        else:
            verts, faces, normals = cache_malhas[nome]

        matriz_mundo = construir_matriz_mundo(px, pz, escala=escala)
        verts_mundo  = aplicar_matriz_mundo(verts, matriz_mundo)

        traces.append(go.Mesh3d(
            x=verts_mundo[:, 0],
            y=verts_mundo[:, 1],
            z=verts_mundo[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=cor_hex,
            flatshading=False,
            lighting=dict(
                ambient=0.4,
                diffuse=0.7,
                specular=0.6,
                roughness=0.4,
                fresnel=0.2,
            ),
            lightposition=dict(x=15, y=30, z=20),
            showscale=False,
            name=label,
            hovertemplate=f"<b>{label}</b><extra></extra>",
        ))

    # --- Layout da cena ---
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text="Peças de Xadrez Modeladas", font=dict(size=20)),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#1a1a2e",
            camera=dict(
                eye=dict(x=0.0, y=1.6, z=2.2),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode="data",
        ),
        paper_bgcolor="#1a1a2e",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        width=900,
        height=800,
    )

    print("Concluído! Abrindo visualização interativa...")
    fig.show()


if __name__ == "__main__":
    main()
