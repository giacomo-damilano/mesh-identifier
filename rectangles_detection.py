import cv2
import numpy as np
from itertools import combinations
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing, psutil
from joblib import Parallel, delayed
from PIL import Image
import pytoshop
from pytoshop.user import nested_layers
import os
from collections import defaultdict

# ===============================
# CONFIG
# ===============================
input_path  = r"C:\\Users\\giaco\\OneDrive\\Desktop\\back_scheme.png"
output_path = r"C:\\Users\\giaco\\OneDrive\\Desktop\\back_scheme_en.png"

dot_radius = 6
line_thickness = 2
dot_color_bgr = (0, 0, 255)
poly_color_bgr = (0, 255, 0)

MAX_SIDE = 175.0
ANGLE_TARGET = 90.0
ANGLE_TOL = 42.5
MAX_TRIPLETS_PER_BATCH = 100

# ===============================
# ADAPTIVE PARALLEL SETTINGS
# ===============================
def get_adaptive_workers():
    try:
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        cores = physical if physical else logical
    except Exception:
        cores = multiprocessing.cpu_count()
    workers = max(1, min(cores - 1, 32))
    print(f"Detected {cores} cores → using {workers} workers")
    return workers

NPROC = get_adaptive_workers()

# ===============================
# GEOMETRY HELPERS
# ===============================
def angle_deg(p1, p2, p3):
    v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cosang))

def order_quad_clockwise(pts):
    c = np.mean(pts, axis=0)
    a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(a)]

def side_lengths(pts):
    return [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def rect_score(pts):
    angs = [abs(angle_deg(pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]) - ANGLE_TARGET) for i in range(4)]
    L = side_lengths(pts)
    angle_penalty = np.mean(angs)
    opp_diff = abs(L[0] - L[2]) + abs(L[1] - L[3])
    aspect_skew = abs((L[0] + L[2]) - (L[1] + L[3]))
    return angle_penalty + 0.5 * opp_diff + 0.25 * aspect_skew

# ===============================
# 1) LOAD IMAGE & DETECT DOTS
# ===============================
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"Could not read: {input_path}")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.bitwise_or(
    cv2.inRange(hsv, np.array([0, 80, 60]), np.array([10, 255, 255])),
    cv2.inRange(hsv, np.array([170, 80, 60]), np.array([180, 255, 255]))
)
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
centers = [
    (float(centroids[i][0]), float(centroids[i][1]))
    for i in range(1, num_labels)
    if stats[i][2] in range(5, 13)
    and stats[i][3] in range(5, 13)
    and (stats[i][4] >= 0.55 * (stats[i][2] * stats[i][3]) or stats[i][4] >= 25)
]
centers = np.array(centers, dtype=np.float32)
n = len(centers)
print(f"Detected {n} dots.")

# ===============================
# 2) KD-TREE NEIGHBORS (robust)
# ===============================
DIAG_FACTOR = np.sqrt(2) * 1.20
R_NEIGH = MAX_SIDE * DIAG_FACTOR
tree = cKDTree(centers)
neighbor_lists = [tree.query_ball_point(centers[i], R_NEIGH) for i in range(n)]

tasks, seen_trips = [], set()
for i in range(n):
    neigh = neighbor_lists[i]
    if len(neigh) < 2:
        continue
    for (j, k) in combinations(neigh, 2):
        trip = tuple(sorted((i, j, k)))
        if trip not in seen_trips:
            seen_trips.add(trip)
            tasks.append(trip)
print(f"Candidate triplets after pruning & true dedup: {len(tasks):,}")

# ===============================
# 3) PARALLEL CANDIDATE GENERATION (robust + diagnostics)
# ===============================
snap_tree = cKDTree(centers)

def process_triplet_batch(batch, centers):
    out = []
    AREA_TOL = (MAX_SIDE * 0.02)**2
    SIDE_RELAX = 1.35
    POS_TOL = max(14.0, MAX_SIDE * 0.1)
    ANGLE_TOL_LOCAL = min(ANGLE_TOL, 42.5)
    DIAG_RATIO_MIN = 1.06
    ASPECT_MAX = 1.35

    diag = {
        'total':0,'too_small_area':0,'two_short_sides_fail':0,'angle_fail':0,
        'snap_fail':0,'diag_too_short':0,'aspect_fail':0,'side_mean_too_long':0
    }

    for (i, j, k) in batch:
        A, B, C = centers[i], centers[j], centers[k]
        diag['total'] += 1
        if triangle_area(A, B, C) < AREA_TOL:
            diag['too_small_area'] += 1
            continue

        dAB, dBC, dCA = np.linalg.norm(A - B), np.linalg.norm(B - C), np.linalg.norm(C - A)
        dists = np.array([dAB, dBC, dCA])
        if np.sum(dists <= MAX_SIDE * SIDE_RELAX) < 2:
            diag['two_short_sides_fail'] += 1
            continue

        for (angle_center, first, second) in ((B, A, C), (A, B, C), (C, A, B)):
            ang = angle_deg(first, angle_center, second)
            if abs(ang - ANGLE_TARGET) > ANGLE_TOL_LOCAL:
                diag['angle_fail'] += 1
                continue

            D_est = first + second - angle_center
            dist, idx = snap_tree.query(D_est, k=1, distance_upper_bound=POS_TOL)
            if not np.isfinite(dist) or idx >= len(centers) or idx in (i, j, k):
                diag['snap_fail'] += 1
                continue

            pts = centers[[i, j, k, idx]].copy()
            pts = order_quad_clockwise(pts)

            sides = side_lengths(pts)
            mean_side = np.mean(sides)
            diag1, diag2 = np.linalg.norm(pts[0] - pts[2]), np.linalg.norm(pts[1] - pts[3])

            if min(diag1, diag2) < mean_side * DIAG_RATIO_MIN:
                diag['diag_too_short'] += 1
                continue
            aspect = max(sides) / max(1e-6, min(sides))
            if aspect > ASPECT_MAX:
                diag['aspect_fail'] += 1
                continue
            if mean_side > MAX_SIDE * SIDE_RELAX:
                diag['side_mean_too_long'] += 1
                continue

            key = tuple(sorted([i, j, k, idx]))
            area = abs(cv2.contourArea(pts.astype(np.float32)))
            score = rect_score(pts)
            out.append((key, pts, score, area))
    return out, diag

batches = [tasks[i:i + MAX_TRIPLETS_PER_BATCH] for i in range(0, len(tasks), MAX_TRIPLETS_PER_BATCH)]
print(f"Processing {len(batches)} batches (≤{MAX_TRIPLETS_PER_BATCH} triplets each) on {NPROC} workers...")

results = Parallel(n_jobs=NPROC, backend="loky", prefer="processes", verbose=0)(
    delayed(process_triplet_batch)(batch, centers) for batch in tqdm(batches, desc="Finding quads", dynamic_ncols=True)
)

candidates = {}
diag_total = {'total':0,'too_small_area':0,'two_short_sides_fail':0,'angle_fail':0,
              'snap_fail':0,'diag_too_short':0,'aspect_fail':0,'side_mean_too_long':0}

for batch_result, diag in results:
    for k in diag_total:
        diag_total[k] += diag.get(k, 0)
    for key, pts, score, area in batch_result:
        if key not in candidates or score < candidates[key]['score']:
            candidates[key] = {'idx': key, 'pts': pts, 'score': score, 'area': area}

print(f"Unique candidate polygons: {len(candidates)}")
print("Diagnostics (triplet stage):", diag_total)

# ===============================
# 4) MAXIMIZE COUNT (non-overlap selection)
# ===============================
print("Building conflict graph (edges allowed, area-overlap forbidden)...")

AREA_EPS = 0.5
CELL_SIZE = max(8, int(MAX_SIDE // 4))

def as_convex_cw(pts):
    hull = cv2.convexHull(pts.astype(np.float32))
    return hull.reshape(-1, 2).astype(np.float32)

cand_list = []
for idx, c in enumerate(candidates.values()):
    poly = as_convex_cw(c['pts'])
    x0, y0 = np.min(poly, axis=0)
    x1, y1 = np.max(poly, axis=0)
    cand_list.append({'idx': idx,'key': c['idx'],'pts': poly,'bbox': (x0, y0, x1, y1),
                      'score': c['score'],'area': c['area']})

M = len(cand_list)
print(f"Candidates to graph: {M}")

grid = defaultdict(list)
def cells_for_bbox(bbox):
    x0, y0, x1, y1 = bbox
    for cx in range(int(x0 // CELL_SIZE), int(x1 // CELL_SIZE) + 1):
        for cy in range(int(y0 // CELL_SIZE), int(y1 // CELL_SIZE) + 1):
            yield (cx, cy)

for i, c in enumerate(cand_list):
    for cell in cells_for_bbox(c['bbox']):
        grid[cell].append(i)

neighbors = [set() for _ in range(M)]
checked = 0
def bboxes_overlap(b1, b2):
    x0a, y0a, x1a, y1a = b1
    x0b, y0b, x1b, y1b = b2
    return not (x1a <= x0b or x1b <= x0a or y1a <= y0b or y1b <= y0a)

for cell, idxs in grid.items():
    idxs = sorted(idxs)
    for a in range(len(idxs)):
        i = idxs[a]
        bi, pi = cand_list[i]['bbox'], cand_list[i]['pts']
        for b in range(a + 1, len(idxs)):
            j = idxs[b]
            bj = cand_list[j]['bbox']
            if not bboxes_overlap(bi, bj): continue
            pj = cand_list[j]['pts']
            area, _ = cv2.intersectConvexConvex(pi, pj)
            if area > AREA_EPS:
                neighbors[i].add(j)
                neighbors[j].add(i)
        checked += 1
print(f"Built conflict graph (approx checks: {checked})")

deg = np.array([len(neighbors[i]) for i in range(M)], dtype=np.int32)
order = np.argsort(deg)
selected_mask = np.zeros(M, dtype=bool)
banned = np.zeros(M, dtype=bool)

for i in tqdm(order, desc="Selecting max-count set", dynamic_ncols=True):
    if banned[i]:
        continue
    selected_mask[i] = True
    for j in neighbors[i]:
        banned[j] = True

selected_indices = set(np.where(selected_mask)[0])
selected = [c for c in cand_list if c['idx'] in selected_indices]
print(f"Selected polygons (max-count heuristic): {len(selected)}")

# ===============================
# 5) DRAW RESULT
# ===============================
out = img.copy()
for c in selected:
    cv2.polylines(out, [c['pts'].astype(np.int32)], True, poly_color_bgr, line_thickness)
for (cx, cy) in centers:
    cv2.circle(out, (int(cx), int(cy)), dot_radius, dot_color_bgr, -1)
cv2.imwrite(output_path, out)
print(f"Saved flattened image: {output_path}")

# ===============================
# 6) EXPORT MULTI-LAYER PSD
# ===============================
def cv2_to_rgba_array(cv_img):
    rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)
    return rgba.astype(np.uint8)

background_layer = cv2_to_rgba_array(img)
dots_layer_bgr = np.zeros_like(img, dtype=np.uint8)
for (cx, cy) in centers:
    cv2.circle(dots_layer_bgr, (int(cx), int(cy)), dot_radius, dot_color_bgr, -1)
dots_layer = cv2_to_rgba_array(dots_layer_bgr)

lines_layer_bgr = np.zeros_like(img, dtype=np.uint8)
for c in selected:
    cv2.polylines(lines_layer_bgr, [c['pts'].astype(np.int32)], True, poly_color_bgr, line_thickness)
lines_layer = cv2_to_rgba_array(lines_layer_bgr)

for arr, name in [(background_layer, "Background"), (lines_layer, "Lines"), (dots_layer, "Dots")]:
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"{name} layer not RGBA! shape={arr.shape}, dtype={arr.dtype}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} layer must be uint8, not {arr.dtype}")

layers = [
    nested_layers.Image(name="Background", image=background_layer),
    nested_layers.Image(name="Lines", image=lines_layer),
    nested_layers.Image(name="Dots", image=dots_layer),
]

psd = nested_layers.nested_layers_to_psd(layers)
psd_path = output_path.replace(".png", "_layers.psd")

try:
    with open(psd_path, "wb") as f:
        psd.write(f)
    if not os.path.exists(psd_path) or os.path.getsize(psd_path) < 1024:
        raise IOError("PSD file not written correctly or empty.")
    print(f"✅ Saved layered PSD: {psd_path}")
except Exception as e:
    print(f"⚠️ PSD export failed: {e}")
