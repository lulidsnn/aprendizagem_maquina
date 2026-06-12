"""
Questão 1: K-médias (Euclidiana e Mahalanobis) com índice Davies-Bouldin
Questão 2: PCA próprio com variância explicada
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# =============================================================================
# UTILITÁRIOS GERAIS
# =============================================================================

def normalizar(X):
    """Normalização Z-score (média 0, desvio padrão 1)."""
    media = X.mean(axis=0)
    desvio = X.std(axis=0)
    return (X - media) / desvio, media, desvio


# =============================================================================
# QUESTÃO 1a — K-MÉDIAS COM DISTÂNCIA EUCLIDIANA
# =============================================================================

def inicializar_centroides(X, k, rng):
    """Seleciona k pontos aleatórios de X como centroides iniciais."""
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def atribuir_clusters_euclidiano(X, centroides):
    """Atribui cada ponto ao centroide mais próximo (distância Euclidiana)."""
    # ||x - c||^2 via broadcasting
    diff = X[:, np.newaxis, :] - centroides[np.newaxis, :, :]   # (n, k, d)
    dists2 = (diff ** 2).sum(axis=2)                             # (n, k)
    return np.argmin(dists2, axis=1)                             # (n,)


def atualizar_centroides(X, rotulos, k):
    """Recalcula centroides como média dos pontos de cada cluster."""
    d = X.shape[1]
    centroides = np.zeros((k, d))
    for j in range(k):
        mask = rotulos == j
        if mask.sum() > 0:
            centroides[j] = X[mask].mean(axis=0)
    return centroides


def erro_reconstrucao_euclidiano(X, centroides, rotulos):
    """SSE — soma dos quadrados das distâncias ao centroide do cluster."""
    sse = 0.0
    for j in range(len(centroides)):
        mask = rotulos == j
        if mask.sum() > 0:
            diff = X[mask] - centroides[j]
            sse += (diff ** 2).sum()
    return sse


def kmeans_euclidiano(X, k, max_iter=300, tol=1e-6, seed=None):
    """Uma execução do K-médias Euclidiano. Retorna (centroides, rotulos, sse)."""
    rng = np.random.default_rng(seed)
    centroides = inicializar_centroides(X, k, rng)
    rotulos = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        novos_rotulos = atribuir_clusters_euclidiano(X, centroides)
        novos_centroides = atualizar_centroides(X, novos_rotulos, k)

        if np.allclose(centroides, novos_centroides, atol=tol):
            rotulos = novos_rotulos
            centroides = novos_centroides
            break

        rotulos = novos_rotulos
        centroides = novos_centroides

    sse = erro_reconstrucao_euclidiano(X, centroides, rotulos)
    return centroides, rotulos, sse


def kmeans_euclidiano_multiplas(X, k, n_repeticoes=20):
    """Repete K-médias n vezes e retorna a melhor solução (menor SSE)."""
    melhor_sse = np.inf
    melhor_centroides = None
    melhor_rotulos = None
    for i in range(n_repeticoes):
        c, r, sse = kmeans_euclidiano(X, k, seed=i)
        if sse < melhor_sse:
            melhor_sse = sse
            melhor_centroides = c
            melhor_rotulos = r
    return melhor_centroides, melhor_rotulos, melhor_sse


# =============================================================================
# QUESTÃO 1a — ÍNDICE DAVIES-BOULDIN EUCLIDIANO
# =============================================================================

def dispersao_euclidiana(X, centroides, rotulos, j):
    """Dispersão média intra-cluster j (distância Euclidiana ao centroide)."""
    mask = rotulos == j
    if mask.sum() == 0:
        return 0.0
    diff = X[mask] - centroides[j]
    return np.sqrt((diff ** 2).sum(axis=1)).mean()


def db_index_euclidiano(X, centroides, rotulos):
    """Índice Davies-Bouldin com distância Euclidiana."""
    k = len(centroides)
    s = np.array([dispersao_euclidiana(X, centroides, rotulos, j) for j in range(k)])

    db_soma = 0.0
    for i in range(k):
        max_r = -np.inf
        for j in range(k):
            if i == j:
                continue
            dist_ij = np.sqrt(((centroides[i] - centroides[j]) ** 2).sum())
            if dist_ij == 0:
                continue
            r_ij = (s[i] + s[j]) / dist_ij
            if r_ij > max_r:
                max_r = r_ij
        db_soma += max_r

    return db_soma / k


# =============================================================================
# QUESTÃO 1b — K-MÉDIAS COM DISTÂNCIA DE MAHALANOBIS
# =============================================================================

def distancia_mahalanobis_batch(X, centroides, VI):
    """
    Calcula distâncias de Mahalanobis de todos os pontos a todos os centroides.
    VI: inversa da matriz de covariância (d x d).
    Retorna matriz (n, k).
    """
    n, k = len(X), len(centroides)
    dists = np.zeros((n, k))
    for j in range(k):
        diff = X - centroides[j]          # (n, d)
        # d_M^2 = diff @ VI @ diff.T (diagonal)
        dists[:, j] = np.einsum('ni,ij,nj->n', diff, VI, diff)
    return np.sqrt(np.maximum(dists, 0))  # (n, k)


def atribuir_clusters_mahalanobis(X, centroides, VI):
    dists = distancia_mahalanobis_batch(X, centroides, VI)
    return np.argmin(dists, axis=1)


def erro_reconstrucao_mahalanobis(X, centroides, rotulos, VI):
    soma = 0.0
    for j in range(len(centroides)):
        mask = rotulos == j
        if mask.sum() > 0:
            diff = X[mask] - centroides[j]
            d2 = np.einsum('ni,ij,nj->n', diff, VI, diff)
            soma += np.sqrt(np.maximum(d2, 0)).sum()
    return soma


def kmeans_mahalanobis(X, k, VI, max_iter=300, tol=1e-6, seed=None):
    """Uma execução do K-médias com distância de Mahalanobis."""
    rng = np.random.default_rng(seed)
    centroides = inicializar_centroides(X, k, rng)
    rotulos = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        novos_rotulos = atribuir_clusters_mahalanobis(X, centroides, VI)
        novos_centroides = atualizar_centroides(X, novos_rotulos, k)

        if np.allclose(centroides, novos_centroides, atol=tol):
            rotulos = novos_rotulos
            centroides = novos_centroides
            break

        rotulos = novos_rotulos
        centroides = novos_centroides

    erro = erro_reconstrucao_mahalanobis(X, centroides, rotulos, VI)
    return centroides, rotulos, erro


def kmeans_mahalanobis_multiplas(X, k, VI, n_repeticoes=20):
    melhor_erro = np.inf
    melhor_centroides = None
    melhor_rotulos = None
    for i in range(n_repeticoes):
        c, r, e = kmeans_mahalanobis(X, k, VI, seed=i)
        if e < melhor_erro:
            melhor_erro = e
            melhor_centroides = c
            melhor_rotulos = r
    return melhor_centroides, melhor_rotulos, melhor_erro


# =============================================================================
# QUESTÃO 1b — ÍNDICE DAVIES-BOULDIN MAHALANOBIS
# =============================================================================

def dispersao_mahalanobis_cluster(X, centroides, rotulos, j, VI):
    """Dispersão média intra-cluster j (distância de Mahalanobis ao centroide)."""
    mask = rotulos == j
    if mask.sum() == 0:
        return 0.0
    diff = X[mask] - centroides[j]
    d2 = np.einsum('ni,ij,nj->n', diff, VI, diff)
    return np.sqrt(np.maximum(d2, 0)).mean()


def dist_mahalanobis_pontos(a, b, VI):
    """Distância de Mahalanobis entre dois vetores."""
    diff = a - b
    return np.sqrt(max(diff @ VI @ diff, 0))


def db_index_mahalanobis(X, centroides, rotulos, VI):
    """Índice Davies-Bouldin com distância de Mahalanobis."""
    k = len(centroides)
    s = np.array([dispersao_mahalanobis_cluster(X, centroides, rotulos, j, VI)
                  for j in range(k)])

    db_soma = 0.0
    for i in range(k):
        max_r = -np.inf
        for j in range(k):
            if i == j:
                continue
            dist_ij = dist_mahalanobis_pontos(centroides[i], centroides[j], VI)
            if dist_ij == 0:
                continue
            r_ij = (s[i] + s[j]) / dist_ij
            if r_ij > max_r:
                max_r = r_ij
        db_soma += max_r

    return db_soma / k


# =============================================================================
# QUESTÃO 2 — PCA PRÓPRIO COM SVD
# =============================================================================

def pca_fit(X_norm, n_componentes=None):
    """
    PCA via SVD (implementação própria).
    X_norm: dados já normalizados, shape (n, d).
    Retorna: componentes (d x d), variância explicada (d,).
    """
    n = X_norm.shape[0]
    # Matriz de dados centrada (já está centralizada após z-score)
    # SVD da matriz de dados (não da covariância diretamente)
    U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)

    # Variância explicada por cada componente
    variancia = (S ** 2) / (n - 1)
    variancia_total = variancia.sum()
    variancia_explicada = variancia / variancia_total   # proporção

    componentes = Vt  # (d, d) — linhas são os componentes principais

    if n_componentes is not None:
        componentes = componentes[:n_componentes]

    return componentes, variancia_explicada


def pca_transform(X_norm, componentes):
    """Projeta X nos componentes principais."""
    return X_norm @ componentes.T   # (n, n_comp)


# =============================================================================
# PLOTAGEM — QUESTÃO 1a
# =============================================================================

def plotar_resultado_q1a(X_orig, X_norm, rotulos, centroides, k, db_values, k_range):
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    cores = plt.cm.tab20(np.linspace(0, 1, k))

    # Painel esquerdo: curva DB
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(k_range, db_values, 'o-', color='steelblue', linewidth=2,
             markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    ax0.axvline(k, color='tomato', linestyle='--', linewidth=1.5,
                label=f'Melhor k={k} (DB={db_values[k - k_range[0]]:.3f})')
    ax0.set_xlabel('Número de Clusters (k)', fontsize=11)
    ax0.set_ylabel('Índice Davies-Bouldin', fontsize=11)
    ax0.set_title('Índice DB × k\n(Euclidiana)', fontsize=12, fontweight='bold')
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax0.set_xticks(k_range)

    # Painel direito: agrupamento em espaço original (lat/lon)
    ax1 = fig.add_subplot(gs[1])
    for j in range(k):
        mask = rotulos == j
        ax1.scatter(X_orig[mask, 1], X_orig[mask, 0],
                    color=cores[j], s=12, alpha=0.55, label=f'Cluster {j+1}')
        # Centroide no espaço original (desnormalizar não é necessário pois
        # usamos os dados normalizados, mas plotamos em coordenadas originais
        # via mapeamento inverso — aqui plotamos apenas posição relativa)

    ax1.set_xlabel('Longitude', fontsize=11)
    ax1.set_ylabel('Latitude', fontsize=11)
    ax1.set_title(f'K-Médias Euclidiana (k={k})\nDados: quake.csv', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7, ncol=2, markerscale=1.5)
    ax1.grid(True, alpha=0.3)

    fig.suptitle('Questão 1a — K-Médias com Distância Euclidiana', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.savefig('/mnt/user-data/outputs/q1a_kmeans_euclidiano.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  [1a] Figura salva.")


# =============================================================================
# PLOTAGEM — QUESTÃO 1b
# =============================================================================

def plotar_resultado_q1b(X_orig, rotulos, centroides_norm, k, db_values, k_range):
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    cores = plt.cm.tab20(np.linspace(0, 1, k))

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(k_range, db_values, 's-', color='darkorchid', linewidth=2,
             markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    ax0.axvline(k, color='tomato', linestyle='--', linewidth=1.5,
                label=f'Melhor k={k} (DB={db_values[k - k_range[0]]:.3f})')
    ax0.set_xlabel('Número de Clusters (k)', fontsize=11)
    ax0.set_ylabel('Índice Davies-Bouldin (Mahalanobis)', fontsize=11)
    ax0.set_title('Índice DB × k\n(Mahalanobis)', fontsize=12, fontweight='bold')
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax0.set_xticks(k_range)

    ax1 = fig.add_subplot(gs[1])
    for j in range(k):
        mask = rotulos == j
        ax1.scatter(X_orig[mask, 1], X_orig[mask, 0],
                    color=cores[j], s=12, alpha=0.55, label=f'Cluster {j+1}')

    ax1.set_xlabel('Longitude', fontsize=11)
    ax1.set_ylabel('Latitude', fontsize=11)
    ax1.set_title(f'K-Médias Mahalanobis (k={k})\nDados: quake.csv', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7, ncol=2, markerscale=1.5)
    ax1.grid(True, alpha=0.3)

    fig.suptitle('Questão 1b — K-Médias com Distância de Mahalanobis', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.savefig('/mnt/user-data/outputs/q1b_kmeans_mahalanobis.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  [1b] Figura salva.")


# =============================================================================
# PLOTAGEM — QUESTÃO 2
# =============================================================================

CORES_ESPECIES = {'Adelie': '#4E79A7', 'Chinstrap': '#F28E2B', 'Gentoo': '#59A14F'}

def plotar_pca_2d(X_proj, especies):
    fig, ax = plt.subplots(figsize=(8, 6))

    for esp, cor in CORES_ESPECIES.items():
        mask = especies == esp
        ax.scatter(X_proj[mask, 0], X_proj[mask, 1],
                   color=cor, s=50, alpha=0.75, edgecolors='white',
                   linewidth=0.4, label=esp)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Projeção PCA em 2D\nPinguins da Antártida', fontsize=13, fontweight='bold')
    ax.legend(title='Espécie', fontsize=10, title_fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/q2a_pca_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [2a] Figura PCA 2D salva.")


def plotar_variancia_explicada(variancia_explicada):
    dims = np.arange(1, len(variancia_explicada) + 1)
    var_acum = np.cumsum(variancia_explicada)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Questão 2b — Variância Explicada pelo PCA', fontsize=13,
                 fontweight='bold')

    # Variância por componente
    bars = axes[0].bar(dims, variancia_explicada * 100,
                       color=['#4E79A7', '#F28E2B', '#59A14F', '#E15759'],
                       edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, variancia_explicada * 100):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f'{val:.1f}%',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Componente Principal', fontsize=11)
    axes[0].set_ylabel('Variância Explicada (%)', fontsize=11)
    axes[0].set_title('Variância por Componente', fontsize=11)
    axes[0].set_xticks(dims)
    axes[0].set_ylim(0, max(variancia_explicada * 100) * 1.15)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Variância acumulada
    axes[1].plot(dims, var_acum * 100, 'o-', color='steelblue',
                 linewidth=2.5, markersize=9, markerfacecolor='white',
                 markeredgewidth=2)
    for d, v in zip(dims, var_acum * 100):
        axes[1].annotate(f'{v:.1f}%', (d, v),
                         textcoords='offset points', xytext=(6, 4),
                         fontsize=10, color='steelblue', fontweight='bold')
    axes[1].axhline(95, color='tomato', linestyle='--', linewidth=1.2,
                    alpha=0.7, label='95%')
    axes[1].set_xlabel('Número de Componentes', fontsize=11)
    axes[1].set_ylabel('Variância Acumulada (%)', fontsize=11)
    axes[1].set_title('Variância Acumulada', fontsize=11)
    axes[1].set_xticks(dims)
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/q2b_variancia_explicada.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [2b] Figura variância salva.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import os
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)

    # -------------------------------------------------------------------------
    # QUESTÃO 1 — Carrega e normaliza quake.csv
    # -------------------------------------------------------------------------
    print("\n=== QUESTÃO 1 — K-MÉDIAS EM QUAKE.CSV ===")
    df_quake = pd.read_csv('/home/claude/quake.csv')
    X_quake_orig = df_quake[['latitude', 'longitude']].values.astype(float)
    X_quake, _, _ = normalizar(X_quake_orig)

    k_range = list(range(4, 21))   # 4, 5, ..., 20

    # ---- 1a: Euclidiana ----
    print("\n[1a] Avaliando DB Euclidiano para k = 4..20 (20 repetições cada)...")
    db_euc = []
    for k in k_range:
        c, r, _ = kmeans_euclidiano_multiplas(X_quake, k, n_repeticoes=20)
        db = db_index_euclidiano(X_quake, c, r)
        db_euc.append(db)
        print(f"  k={k:2d}  DB(Eucl)={db:.4f}")

    k_euc = k_range[int(np.argmin(db_euc))]
    print(f"\n  → Melhor k (Euclidiana): {k_euc}  (DB={min(db_euc):.4f})")
    c_euc, r_euc, sse_euc = kmeans_euclidiano_multiplas(X_quake, k_euc, n_repeticoes=20)
    plotar_resultado_q1a(X_quake_orig, X_quake, r_euc, c_euc, k_euc, db_euc, k_range)

    # ---- 1b: Mahalanobis ----
    print("\n[1b] Calculando matriz de covariância global para Mahalanobis...")
    Sigma = np.cov(X_quake.T)
    VI = np.linalg.inv(Sigma)
    print(f"  Sigma:\n{Sigma}")
    print(f"  Sigma^-1:\n{VI}")

    print("\n[1b] Avaliando DB Mahalanobis para k = 4..20 (20 repetições cada)...")
    db_mah = []
    for k in k_range:
        c, r, _ = kmeans_mahalanobis_multiplas(X_quake, k, VI, n_repeticoes=20)
        db = db_index_mahalanobis(X_quake, c, r, VI)
        db_mah.append(db)
        print(f"  k={k:2d}  DB(Mahal)={db:.4f}")

    k_mah = k_range[int(np.argmin(db_mah))]
    print(f"\n  → Melhor k (Mahalanobis): {k_mah}  (DB={min(db_mah):.4f})")
    c_mah, r_mah, err_mah = kmeans_mahalanobis_multiplas(X_quake, k_mah, VI, n_repeticoes=20)
    plotar_resultado_q1b(X_quake_orig, r_mah, c_mah, k_mah, db_mah, k_range)

    # -------------------------------------------------------------------------
    # QUESTÃO 2 — PCA em penguins.csv
    # -------------------------------------------------------------------------
    print("\n=== QUESTÃO 2 — PCA EM PENGUINS.CSV ===")
    df_peng = pd.read_csv('/home/claude/penguins.csv')
    especies = df_peng['species'].values
    X_peng_orig = df_peng[['bill_length_mm','bill_depth_mm',
                            'flipper_length_mm','body_mass_g']].values.astype(float)

    X_peng, _, _ = normalizar(X_peng_orig)

    # PCA completo (d=4 componentes)
    componentes, var_exp = pca_fit(X_peng)
    print(f"\n  Variância explicada por componente:")
    for i, v in enumerate(var_exp):
        print(f"    PC{i+1}: {v*100:.2f}%  (acumulada: {var_exp[:i+1].sum()*100:.2f}%)")

    # 2a: projeção 2D
    X_2d = pca_transform(X_peng, componentes[:2])
    plotar_pca_2d(X_2d, especies)

    # 2b: variância por número de componentes
    print("\n  Variância acumulada por número de componentes:")
    for d in range(1, 5):
        acum = var_exp[:d].sum()
        print(f"    {d} componente(s): {acum*100:.2f}%")
    plotar_variancia_explicada(var_exp)

    print("\n✔ Todas as figuras salvas em /mnt/user-data/outputs/")