from __future__ import annotations
from typing import Dict
import re
import numpy as np
import math


# =========================
# Segment 1: String -> Vector (1/2/3 contiguous n-grams)
# =========================

def _clean_letters(s: str) -> str:
    """Lowercase and keep only a-z letters."""
    return re.sub(r"[^a-z]", "", s.lower())


def string_to_vector(s: str) -> Dict[str, float]:
    """
    Convert a string into contiguous character n-gram counts for n = 1, 2, 3.

    Preprocessing requirements:
      - Convert to lowercase
      - Keep only letters a–z
      - Generate contiguous character n-grams for n in {1,2,3}
      - Count occurrences (as float values)

    Returns:
      A dictionary mapping n-gram (str) -> count (float)
    """
    # TODO (Segment 1): implement contiguous 1/2/3-gram counts
    s = _clean_letters(s)
    str_vectors = {}
    #loop through and (if not last 2/3 chars), process character strings of length 1,2,3
    #add to dict unless it's already in the dict (in which case you increment)
    for i in range(s.len()):
      if (s[i] in str_vectors.keys()):
        str_vectors[s[i]] += 1
      else:
        str_vectors.update({s[i]: 1})
      
      if i < (s.len() - 1):
        continue
      string_bit = s[i] + s[i + 1]
      if (string_bit in str_vectors.keys()):
        str_vectors[string_bit] += 1
      else:
        str_vectors.update({string_bit: 1})

      if i < (s.len() - 2):
        continue
      string_bit = s[i] + s[i + 1] + s[i + 2]
      if (string_bit in str_vectors.keys()):
        str_vectors[string_bit] += 1
      else:
        str_vectors.update({string_bit: 1})

    return str_vectors

def l1_distance(v1, v2) -> float:
    """
    Compute L1 distance between two vectors.

    Inputs may be either:
      - Sparse vectors (dict[str, float])
      - Dense vectors (lists or numpy arrays)

    Assumptions:
      - Both inputs are of the same type
      - For sparse vectors, missing keys are treated as 0

    Returns:
      L1 distance (float)
    """
    # TODO (Segment 2): implement L1 distance for both sparse and dense cases
    #sparse
    norm = 0.0
    if type(v1) == Dict[str, float]:
       #loop thru keys in v1 and compare to v2 (pure difference)
       for key in v1:
          if key in v2:
             norm += math.abs(v1[key] - v2[key])
          else:
             norm += math.abs(v1[key])
       #loop thru keys in v2, discard if found in v1, add distance (set to 0) if not
       for key in v2:
          if not key in v1:
             norm += math.abs(v2[key])
    else:
       for i in range(len(v1)):
          norm += math.abs(v1[i] - v2[i])
    return norm


def l2_distance(v1, v2) -> float:
    """
    Compute L2 distance between two vectors.

    Inputs may be either:
      - Sparse vectors (dict[str, float])
      - Dense vectors (lists or numpy arrays)

    Assumptions:
      - Both inputs are of the same type
      - For sparse vectors, missing keys are treated as 0

    Returns:
      L2 distance (float)
    """
    # TODO 
    norm = 0.0
    if type(v1) == Dict[str, float]:
       #loop thru keys in v1 and compare to v2 (pure difference)
       for key in v1:
          if key in v2:
             norm += math.pow(v1[key] - v2[key], 2)
          else:
             norm += math.pow(v1[key], 2)
       #loop thru keys in v2, discard if found in v1, add distance (set to 0) if not
       for key in v2:
          if not key in v1:
             norm += math.pow(v2[key], 2)
    else:
       for i in range(len(v1)):
          norm += math.pow(v1[i] - v2[i], 2)
    return norm


def update_centroids(X, assignments, k):
    """
    Perform the centroid update step in k-means clustering.

    Inputs:
      - X: (n x d) dense numpy array of data points
      - assignments: length-n array of cluster indices in {0, ..., k-1}
      - k: number of clusters

    Returns:
      - centroids: (k x d) numpy array where row j is the mean of
        all points assigned to cluster j.

    Requirement:
      - If a cluster j has no assigned points, its centroid must be
        the all-zeros vector of dimension d.
    """
    # TODO: compute cluster means and handle empty clusters

    # list of k d-dimensional arrays for sums of positions of points
    # list of k ints counting no. of data points in a cluster
    d = len[X[0]]
    centroids = np.zeros((k, d))
    cluster_counts = np.zeros((k))

    for i in range(len(assignments)):
       centroids[assignments[i]] += X[i]
       cluster_counts[assignments[i]] += 1

    # averaging function to find centroids (account for empty cluster case)
    for i in range(len(cluster_counts)):
      if cluster_counts[i] != 0:
        centroids[i] / cluster_counts[i]
    return centroids


def assign_clusters(X, centroids):
    """
    Assign each point in X to its nearest centroid using L2 distance.

    Inputs:
      - X: (n x d) array-like
      - centroids: (k x d) array-like

    Returns:
      - assignments: length-n integer numpy array
        assignments[i] is the index of the closest centroid.

    Tie-breaking rule:
      - If distances are equal, choose the smaller centroid index.

    Hint:
      - You may reuse the l2_distance function defined above.
    """
    # TODO: compute nearest centroid using L2 distance

    # Loop thru each point in X, 
    assignments = np.zeros([len(X)])
    for i in range(len(X)):
      smallest_dist = math.inf
      for j in range(len(centroids)):
        dist = l2_distance(X[i], centroids[j])
        if dist < smallest_dist:
           smallest_dist = dist
           assignments[i] = j

    return assignments

def svd_symmetric_embeddings(X, k, eps=1e-12):
    """
    Segment 5: Word embeddings from a symmetric co-occurrence matrix.

    Given:
      - X: (V x V) real symmetric matrix
      - k: desired embedding dimension (k <= V)

    Goal:
      - Return an embedding matrix E of shape (V x k) computed from a rank-k
        spectral factorization of X.

    Autograder standardization requirement:
      - You MUST use this exact call (do not change it):
          evals, evecs = np.linalg.eigh(X)

    Output requirements:
      - E must be real-valued and finite
      - E must have shape (V x k)

    Hint:
      - Numerical issues can produce tiny negative eigenvalues; handle safely.

    Returns:
      - E: (V x k) numpy array
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square (V x V) matrix.")
    V = X.shape[0]
    if not (1 <= k <= V):
        raise ValueError("k must satisfy 1 <= k <= V.")

    # REQUIRED standardized call
    evals, evecs = np.linalg.eigh(X)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    #evals - 1D array of the eigenvalues of X
    #evecs - eigenvectors (each column is an eigenvector)

    # TODO use svd_symmetric_embeddings(X, k, eps) to construct and return a rank-k approximation of X
    # use epsilon (round values to closest epsilon if not handled later)
    # U_k are the first k columns of U, Sigma_k are the first k singular values, and V_k are the first k rows of V
    # Note that the SVD for a symmetric matrix means U is unitary, so A = U(Sigma)U^T
    evals = np.round(evals / eps) * eps
    #edit such that evals is in descending order
    U = evecs[:, :k]
    evals_k = evals[:k]
    rooted_evals = np.emath.sqrt(evals_k)
    Sigma = np.diag(rooted_evals)
    embedded = U @ Sigma
    return embedded


def rank_k_approx_symmetric(X, k, eps=1e-12):
    """
    Construct a rank-k approximation of a symmetric matrix.

    Requirements:
      - Output must have shape (V x V)
      - Output should be symmetric up to numerical precision
      - Use your svd_symmetric_embeddings implementation

    Returns:
      - X_k: (V x V) numpy array
    """
    # TODO: compute E = svd_symmetric_embeddings(X, k, eps) and return X_k from E
    E_mat = svd_symmetric_embeddings(X, k, eps)
    E_trans = np.transpose(E_mat)
    X_k = E_mat @ E_trans
    return X_k

def svd_unsymmetric_embeddings(X, k):
    """
    Segment 6: Collaborative filtering via truncated SVD.

    Given:
      - X: (M x N) real-valued user–item matrix
      - k: desired latent dimension (k <= min(M, N))

    Goal:
      - Return two embedding matrices:
          U_emb: (M x k)
          M_emb: (N x k)

    Autograder standardization requirement:
      - You MUST use this exact call (do not change it):
          U, S, Vh = np.linalg.svd(X, full_matrices=False)

    Output requirements:
      - Use only the top-k components
      - Outputs must be real-valued and finite
      - Shapes must match (M x k) and (N x k)

    Returns:
      - U_emb, M_emb
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    M, N = X.shape
    if not (1 <= k <= min(M, N)):
        raise ValueError("k must satisfy 1 <= k <= min(M, N).")

    # REQUIRED standardized call
    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    # TODO (Segment 6): construct and return U_emb and M_emb
    U = np.resize(U, (len(U), k))
    V = np.transpose(np.resize(Vh, (k, len(Vh[0]))))
    Sigma = np.diag(np.sqrt(np.resize(S, k)))
    U_emb = U @ Sigma
    M_emb = V @ Sigma
    return U_emb, M_emb


def rank_k_approx_unsymmetric(X, k):
    """
    Construct a rank-k approximation of X using your embedding function.

    Requirements:
      - Output must have shape (M x N)
      - Use svd_unsymmetric_embeddings
      - Result should match X up to rank-k truncation

    Returns:
      - X_k: (M x N) numpy array
    """
    # TODO: use svd_unsymmetric_embeddings(X, k) to construct and return a rank-k approximation of X
    U_emb, M_emb = svd_unsymmetric_embeddings(X, k)
    V_mult = np.transpose(M_emb)
    return U_emb @ V_mult

