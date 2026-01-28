import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Polytopes (geometry layer)
# -------------------------

def tesseract_vertices():
    # 16 vertices: all sign choices in R^4
    return np.array([[s0, s1, s2, s3]
                     for s0 in (-1, 1)
                     for s1 in (-1, 1)
                     for s2 in (-1, 1)
                     for s3 in (-1, 1)], dtype=float)

def tesseract_edges(V):
    # edge between vertices that differ in exactly one coordinate
    edges = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if np.sum(V[i] != V[j]) == 1:
                edges.append((i, j))
    return edges

def simplex4_vertices(scale=1.0):
    """
    Regular 4-simplex (5 vertices) embedded in R^4.

    Construction:
    - Start with standard basis e_i in R^5 (i=0..4)
    - Subtract centroid so points lie in hyperplane sum x_i = 0
    - Drop last coordinate to get R^4 coordinates (still regular up to linear isometry)
    - Optionally rescale.
    """
    E5 = np.eye(5)
    centroid = np.mean(E5, axis=0)
    V5 = E5 - centroid              # 5 points in 4D hyperplane in R^5
    V4 = V5[:, :4]                  # represent that hyperplane in R^4 (dropping one coord)

    # Normalize so average edge length is ~scale (optional but handy)
    # Compute one edge length (all are equal for regular simplex)
    d = np.linalg.norm(V4[0] - V4[1])
    V4 = (scale / d) * V4
    return V4

def simplex_edges(V):
    # Complete graph: all pairs (5 vertices -> 10 edges)
    edges = []
    n = len(V)
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return edges

def make_polytope(kind="tesseract"):
    kind = kind.lower().strip()
    if kind in ("tesseract", "hypercube", "cube4", "4-cube"):
        V = tesseract_vertices()
        E = tesseract_edges(V)
        return V, E
    if kind in ("simplex", "4-simplex", "simplex4", "pentachoron"):
        V = simplex4_vertices(scale=2.0)  # tweak scale to taste
        E = simplex_edges(V)
        return V, E
    raise ValueError(f"Unknown polytope kind: {kind}")

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
    # angles is a dict like { (0,1):a, (2,3):b, (0,2):c, ... }
    R = np.eye(4)
    for (i, j), th in angles.items():
        R = rot_in_plane(4, i, j, th) @ R
    return vs @ R.T

def project_4_to_3(vs4, mode="drop_w", w_scale=0.6):
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

def draw_edges(v2, edges, z, outfile="shape.svg"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    order = sorted(edges, key=lambda e: (z[e[0]] + z[e[1]])/2.0)
    for i, j in order:
        (x1, y1), (x2, y2) = v2[i], v2[j]
        ax.plot([x1, x2], [y1, y2], linewidth=1.6, solid_capstyle="round")

    fig.savefig(outfile, bbox_inches="tight", transparent=True)
    plt.close(fig)

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    polytope = "4-simplex"   # try "4-simplex"
    V, E = make_polytope(polytope)

    angles = {(0,1): 0.7, (2,3): 1.1}
    V4 = rotate4(V, angles)
    V3 = project_4_to_3(V4, mode="w_shear", w_scale=0.75)
    V2, z = project_3_to_2(V3, perspective=True, camera_dist=6.0)

    out = "tesseract.svg" if polytope == "tesseract" else "simplex4.svg"
    draw_edges(V2, E, z, outfile=out)