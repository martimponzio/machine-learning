import networkx as nx
from pathlib import Path


def load_mtx_graph(path: Path) -> nx.DiGraph:
    """
    Lê um arquivo MatrixMarket (.mtx) de grafo e devolve um DiGraph do NetworkX.
    """
    G = nx.DiGraph()

    with path.open("r", encoding="utf-8") as f:
        # pular cabeçalho e comentários (%)
        for line in f:
            if line.startswith("%"):
                continue
            # primeira linha não comentada = nrows ncols nnz
            parts = line.strip().split()
            nrows, ncols, nnz = map(int, parts)
            break

        # resto das linhas: arestas (i, j)
        for line in f:
            if not line.strip():
                continue
            i, j = line.split()[:2]
            u, v = int(i), int(j)
            G.add_edge(u, v)

    return G


#  Implementação manual do PageRank

def pagerank_manual(G: nx.DiGraph, d: float = 0.85,
                    tol: float = 1e-6, max_iter: int = 1000):
    """
    Implementação iterativa do PageRank (random surfer).

    PR_i^{k+1} = (1-d)/N + d * sum_{j in In(i)} PR_j^k / L_j + tratamento de dangling nodes
    """
    nodes = list(G.nodes())
    N = len(nodes)

    # inicialização uniforme
    pr = {node: 1.0 / N for node in nodes}

    # outdegree de cada nó
    outdeg = {node: G.out_degree(node) for node in nodes}

    teleport = (1.0 - d) / N

    for it in range(max_iter):
        new_pr = {}
        # soma da massa de PageRank de nós sem saída
        dangling_sum = sum(pr[n] for n in nodes if outdeg[n] == 0)

        for i in nodes:
            rank = teleport
            # redistribui massa dos dangling nodes igualmente
            rank += d * dangling_sum / N

            # contribuições dos vizinhos de entrada
            for j in G.predecessors(i):
                if outdeg[j] > 0:
                    rank += d * pr[j] / outdeg[j]

            new_pr[i] = rank

        # critério de convergência
        diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
        pr = new_pr

        if diff < tol:
            print(f"[d={d}] Convergiu em {it+1} iterações (diferença={diff:.2e})")
            break
    else:
        print(f"[d={d}] NÃO convergiu em {max_iter} iterações (diferença={diff:.2e})")

    return pr


# Funções auxiliares

def top_k(pr_dict, k=10):
    """Retorna lista (nó, score) ordenada do maior para o menor."""
    return sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:k]


def compare_with_networkx(G, pr_manual, d):
    """Compara PageRank manual com o do NetworkX e imprime diferença máxima."""
    pr_nx = nx.pagerank(G, alpha=d, max_iter=1000, tol=1e-08)
    max_diff = max(abs(pr_manual[n] - pr_nx[n]) for n in G.nodes())
    print(f"[d={d}] Diferença máxima entre manual e NetworkX: {max_diff:.4e}")
    return pr_nx


# main

def main():
    # pasta onde está este arquivo pagerank.py
    base_dir = Path(__file__).resolve().parent

    # agora o arquivo está em Page_rank/source/email-Eu-core.mtx
    data_path = base_dir / "source" / "email-Eu-core.mtx"

    print(f"Carregando grafo de: {data_path}")
    print("Arquivo existe?", data_path.exists())

    G = load_mtx_graph(data_path)
    print(f"Grafo carregado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.\n")

    # valores de d a serem testados
    damping_values = [0.5, 0.85, 0.99]

    for d in damping_values:
        print("=" * 60)
        print(f"Rodando PageRank manual com d = {d} ...")
        pr_manual = pagerank_manual(G, d=d, tol=1e-6, max_iter=1000)

        # comparar com networkx
        pr_nx = compare_with_networkx(G, pr_manual, d)

        # top 10 nós (manual)
        top10 = top_k(pr_manual, k=10)

        print(f"\nTop 10 nós para d = {d} (PageRank manual):")
        for node, score in top10:
            print(f"  Nó {node:4d}  ->  PR = {score:.6f}")

        print()  # linha em branco


if __name__ == "__main__":
    main()
