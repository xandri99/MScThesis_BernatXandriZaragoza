import numpy as np
import scipy.sparse


def coarsen(A, levels, self_connections=False):
    graphs, parents = [A], []

    for _ in range(levels):
        last_graph = graphs[-1]
        N, _ = last_graph.shape
        weights = np.array(last_graph.sum(axis=0)).flatten()  # Using degree as weight
        rr, cc, vv = scipy.sparse.find(last_graph)

        cluster_id = metis_one_level(rr, cc, vv, N)
        parents.append(cluster_id)
        Nnew = np.max(cluster_id) + 1  # Compute new graph size from max cluster ID

        # Rebuild the graph based on new clusters
        new_data = []
        new_row = []
        new_col = []
        for idx in range(len(vv)):
            new_row.append(cluster_id[rr[idx]])
            new_col.append(cluster_id[cc[idx]])
            new_data.append(vv[idx])
        new_graph = scipy.sparse.coo_matrix((new_data, (new_row, new_col)), shape=(Nnew, Nnew)).tocsr()
        new_graph.eliminate_zeros()

        graphs.append(new_graph)

    return graphs, parents

def metis(W, levels, rid=None):
    N, _ = W.shape
    if rid is None:
        rid = np.random.permutation(N)
    parents = []
    degree = W.sum(axis=0).A.flatten() - W.diagonal()
    graphs = [W]

    for _ in range(levels):
        weights = np.array(degree)
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = metis_one_level(rr, cc, vv, rid, weights)
        parents.append(cluster_id)

        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        W = scipy.sparse.csr_matrix((nvv, (nrr, ncc)), shape=(Nnew, Nnew))
        W.eliminate_zeros()
        graphs.append(W)
        N, _ = W.shape
        degree = W.sum(axis=0).A.flatten()

    return graphs, parents

def metis_one_level(rr, cc, vv, N, reduction_factor=2):
    """
    Clusters nodes with an attempt to reduce the graph size by `reduction_factor`.
    """
    # Number of desired clusters is about half of the current number of nodes
    desired_clusters = N // reduction_factor

    cluster_id = np.full(N, -1, dtype=int)
    clustercount = 0

    # Priority is given by edge weights
    sorted_edges = np.argsort(-vv)  # Sort edges by descending weight for prioritized clustering
    for index in sorted_edges:
        i, j = rr[index], cc[index]
        if cluster_id[i] == -1 and cluster_id[j] == -1:
            if clustercount < desired_clusters:
                # Create a new cluster with these two nodes
                cluster_id[i] = clustercount
                cluster_id[j] = clustercount
                clustercount += 1

    # Assign remaining unclustered nodes to their own cluster
    for i in range(N):
        if cluster_id[i] == -1:
            cluster_id[i] = clustercount
            clustercount += 1

    return cluster_id

def compute_perm(parents):
    indices = []
    if parents:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in reversed(parents):
        pool_singeltons = len(parent)
        indices_layer = []

        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            if len(indices_node) == 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
            elif len(indices_node) == 0:
                indices_node.extend([pool_singeltons, pool_singeltons + 1])
                pool_singeltons += 2
            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    return indices[::-1]


def perm_data(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    N, M = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew))
    for i,j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:,i] = x[:,j]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[:,i] = np.zeros(N)
    return xnew

def perm_adjacency(A, indices):
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()

    # Add Mnew - M isolated vertices.
    if Mnew > M:
        rows = scipy.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
        cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])

    # Permute the rows and the columns.
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    # assert np.abs(A - A.T).mean() < 1e-9
    assert type(A) is scipy.sparse.coo.coo_matrix
    return A