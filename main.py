import numpy as np
import matplotlib.pyplot as plt

def tesseract_vertices():
    # 16 vertices: all sign choices in R^4
    vs = np.array([[s0, s1, s2, s3]
                   for s0 in (-1, 1)
                   for s1 in (-1, 1)
                   for s2 in (-1, 1)
                   for s3 in (-1, 1)], dtype=float)
    return vs

def tesseract_edges():
    # edge between vertices that differ in exactly one coordinate
    V = tesseract_vertices()
    edges = []
    for i in range(len(V)):
        for j in range(i+1, len(V)):
            if np.sum(V[i] != V[j]) == 1:
                edges.append((i, j))
    return edges

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
    # a simple "immersion-like" choice: let w influence xyz slightly
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
    # simple perspective
    f = camera_dist / (camera_dist - z)
    return np.c_[f*x, f*y], z

def draw_edges(v2, edges, z, outfile="tesseract.svg"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # simple depth styling: sort edges by average z (far -> near)
    order = sorted(edges, key=lambda e: (z[e[0]] + z[e[1]])/2.0)
    for i, j in order:
        (x1, y1), (x2, y2) = v2[i], v2[j]
        ax.plot([x1, x2], [y1, y2], linewidth=1.6, solid_capstyle="round")

    fig.savefig(outfile, bbox_inches="tight", transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    V = tesseract_vertices()
    E = tesseract_edges()

    angles = {(0,1): 0.6, (2,3): 0.9, (0,2): 0.35, (1,3): 0.25}
    V4 = rotate4(V, angles)
    V3 = project_4_to_3(V4, mode="w_shear", w_scale=0.75)
    V2, z = project_3_to_2(V3, perspective=True, camera_dist=6.0)
    draw_edges(V2, E, z, outfile="tesseract.svg")