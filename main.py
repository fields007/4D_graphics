import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# -------------------------
# Geometry: 4-cube + its simplex triangulation
# -------------------------

def tesseract_vertices01():
    """
    16 vertices of the unit 4-cube [0,1]^4 as binary vectors.
    Order is lexicographic in (x0,x1,x2,x3).
    """
    return np.array([[b0, b1, b2, b3]
                     for b0 in (0, 1)
                     for b1 in (0, 1)
                     for b2 in (0, 1)
                     for b3 in (0, 1)], dtype=float)

def vertex_index_from_bits(bits):
    """
    bits: length-4 iterable of 0/1
    Must match the lexicographic order used in tesseract_vertices01().
    """
    b0, b1, b2, b3 = map(int, bits)
    return (b0 << 3) | (b1 << 2) | (b2 << 1) | b3

def tesseract_edges_outer(V):
    """
    Outer edges of the 4-cube: connect vertices differing in exactly one coordinate.
    Works for either {0,1}^4 or {-1,1}^4 vertices.
    """
    edges = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if np.sum(V[i] != V[j]) == 1:
                edges.append((i, j))
    return edges

def tesseract_triangulation_simplices():
    """
    Canonical triangulation of [0,1]^4 into 4! = 24 congruent 4-simplices.

    For each permutation sigma of {0,1,2,3}, take the 5 vertices:
      v0 = (0,0,0,0)
      v1 = e_{sigma(1)}
      v2 = e_{sigma(1)} + e_{sigma(2)}
      v3 = e_{sigma(1)} + e_{sigma(2)} + e_{sigma(3)}
      v4 = (1,1,1,1)

    Returns a list of simplices, each as a tuple of 5 vertex indices in [0..15].
    """
    simplices = []
    for perm in permutations(range(4)):
        bits = np.zeros(4, dtype=int)
        chain = [tuple(bits)]
        for k in perm:
            bits[k] = 1
            chain.append(tuple(bits))
        simplex = tuple(vertex_index_from_bits(b) for b in chain)  # 5 indices
        simplices.append(simplex)
    return simplices

def edges_from_simplices(simplices):
    """
    Given a list of simplices (each a tuple of vertex indices),
    return the set of all edges appearing in any simplex.
    """
    edge_set = set()
    for s in simplices:
        m = len(s)
        for a in range(m):
            for b in range(a + 1, m):
                i, j = s[a], s[b]
                edge_set.add((i, j) if i < j else (j, i))
    return sorted(edge_set)

# -------------------------
# Rotations / projections
# -------------------------

def rot_in_plane(n, i, j, theta):
    R = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c;  R[i, j] = -s
    R[j, i] = s;  R[j, j] = c
    return R

def rotate4(vs, angles):
    """
    angles: dict like { (0,1):a, (2,3):b, (0,2):c, ... }
    """
    R = np.eye(4)
    for (i, j), th in angles.items():
        R = rot_in_plane(4, i, j, th) @ R
    return vs @ R.T

def project_4_to_3(vs4, mode="w_shear", w_scale=0.6):
    x, y, z, w = vs4.T
    if mode == "drop_w":
        return np.c_[x, y, z]
    if mode == "w_shear":
        return np.c_[x + w_scale*w, y + 0.3*w, z + 0.15*w]
    raise ValueError("unknown mode")

def project_3_to_2(vs3, perspective=False, camera_dist=5.0):
    x, y, z = vs3.T
    if not perspective:
        return np.c_[x, y], z
    f = camera_dist / (camera_dist - z)
    return np.c_[f*x, f*y], z

# -------------------------
# Drawing
# -------------------------

def draw_edges(v2, edges, z, outfile="shape.svg", linewidth=1.6):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # draw far -> near by average depth
    order = sorted(edges, key=lambda e: (z[e[0]] + z[e[1]]) / 2.0)
    for i, j in order:
        (x1, y1), (x2, y2) = v2[i], v2[j]
        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, solid_capstyle="round")

    fig.savefig(outfile, bbox_inches="tight", transparent=True)
    plt.close(fig)

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    # 1) base vertices of the 4-cube
    V = tesseract_vertices01()

    # 2) triangulation into 24 4-simplices
    simplices = tesseract_triangulation_simplices()

    # Choose what edges you want to render:
    # - triangulation edges (denser, shows interior structure)
    E_tri = edges_from_simplices(simplices)

    # - outer cube edges only (classic wireframe)
    E_outer = tesseract_edges_outer(V)

    # Pick one:
    edges_to_draw = E_tri          # change to E_outer if you want only outer edges

    # 3) rotate / project (same as before)
    angles = {(0,1): 0.7, (2,3): 1.1}
    V4 = rotate4(V, angles)

    V3 = project_4_to_3(V4, mode="w_shear", w_scale=0.75)
    V2, z = project_3_to_2(V3, perspective=True, camera_dist=6.0)

    # 4) draw
    draw_edges(V2, edges_to_draw, z, outfile="tesseract_triangulated.svg", linewidth=1.4)