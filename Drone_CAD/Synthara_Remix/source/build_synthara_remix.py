"""
Synthara remix for the sub-250g NLP drone.
Base: Project Synthara generative frame (3.5in, GEPRC Smart35 spacing).
Frame convention after import: Z up, bottom of backbone at z=-6, motor pad tops at z=3.
Front = +Y (MTF-01 pod), rear = -Y (ESP32 cradle).

REMOVED  : raised top plate, camera ears, centre posts/cage, Taker G4 26x26 mount (covered by new plate),
           arm material rising inside the 19.3mm LANNRC bell footprint.
ADDED    : 30.5x30.5 M3 stack plate + clear ESC bay, MTF-01 downward pod, ESP32 rear cradle,
           truss prop guards for 3in props.
IMPROVED : motor bolt holes 1.99 -> 2.2mm clearance (Ø9 bolt circle kept), flat Ø22 motor seats.
"""
import math
import numpy as np
import trimesh
import manifold3d as mf

SRC = "/Users/ianavelino/Downloads/synthara-frame.stl"
OUT = "/Users/ianavelino/Desktop/Claude/Drone_CAD/Synthara_Remix"

SEG = 48
M = mf.Manifold
CS = mf.CrossSection


# ---------------- helpers ----------------
def to_manifold(tm):
    return M(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                     tri_verts=np.asarray(tm.faces, np.uint32)))


def to_trimesh(man):
    mesh = man.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=True)


def box(x0, x1, y0, y1, z0, z1):
    return M.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def cyl(x, y, r, z0, z1, seg=SEG):
    return M.cylinder(z1 - z0, r, circular_segments=seg).translate((x, y, z0))


def ring(x, y, r_in, r_out, z0, z1):
    return cyl(x, y, r_out, z0, z1, 96) - cyl(x, y, r_in, z0 - 1, z1 + 1, 96)


def rounded_rect(w, h, r, z0, z1, cx=0.0, cy=0.0):
    cs = CS.square((w - 2 * r, h - 2 * r), center=True).offset(r, mf.JoinType.Round, circular_segments=32)
    return cs.extrude(z1 - z0).translate((cx, cy, z0))


def strut(p0, p1, r):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    z = d / L
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.column_stack([x, y, z, p0])            # 3x4
    return M.cylinder(L, r, circular_segments=8).transform(T)


# ---------------- load base (Y-up STL -> Z-up, proper rotation, no mirror) ----------------
tm = trimesh.load(SRC)
tm.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]]))
base = to_manifold(tm)
print(f"base volume {base.volume()/1000:.2f} cm3, status {base.status()}")

MOTORS = [(sx * 60.0, sy * 49.0) for sx in (-1, 1) for sy in (-1, 1)]

# detect existing motor bolt holes (slice through the pad)
holes = []
for poly in base.slice(1.5).to_polygons():
    P = np.asarray(poly)
    if len(P) < 6:
        continue
    x, y = P[:, 0], P[:, 1]
    area = abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    c = P.mean(0)
    if 2.0 < area < 4.5 and min(math.hypot(c[0] - mx, c[1] - my) for mx, my in MOTORS) < 6:
        holes.append(c)
holes = np.array(holes)
print(f"detected {len(holes)} motor bolt holes; radius from motor axis:",
      sorted({round(min(math.hypot(h[0]-mx, h[1]-my) for mx, my in MOTORS), 2) for h in holes}))

# ---------------- parameters ----------------
PLATE_Z0, PLATE_Z1 = -6.0, -3.0          # stack plate, same plane as backbone
STACK_SP, M3 = 30.5, 3.2
BAY = (24.0, 23.5)                        # half-extents of ESC keep-out (board 45.6 x 44)
TRIM_Z = 12.5                             # everything above this is old top structure

PROP_R = 38.1
G_IN, G_TOP_OUT, G_BOT_OUT = 40.5, 41.7, 42.3
G_BOT = (-6.0, -3.5)
G_TOP = (17.75, 24.25)                    # blade band: pad 3 + motor 14.8 + hub
N_STRUT = 12

SEAT_R, SEAT_Z0 = 11.0, 3.02             # clears 19.3mm bell (+0.85mm radial)
M2 = 2.2

# MTF-01 pod (front), long axis along Y
POD_IN = (8.8, 26.0, 55.9)                # half-width x, y start, y end (pocket)
POD_WALL, POD_TOP = 1.7, 2.0
MTF_HOLES = [(sx * 6.0, 40.95 + sy * 12.15) for sx in (-1, 1) for sy in (-1, 1)]

# ESP32 DevKitC cradle (rear), long axis along Y
ESP_Y0, ESP_Y1 = -25.0, -81.0
ESP_RAIL = (12.6, 15.4)
ESP_POST_TOP = 6.0                        # PCB underside; header pins end ~8.5mm lower (z -2.5)

# ---------------- REMOVE ----------------
frame = base
frame -= box(-100, 100, -100, 100, TRIM_Z, 60)                                   # top plate, ears, posts
frame -= box(-BAY[0], BAY[0], -BAY[1], BAY[1], PLATE_Z1, TRIM_Z + 1)             # ESC bay
frame -= box(-(POD_IN[0] + 0.5), POD_IN[0] + 0.5, POD_IN[1] - 0.5, POD_IN[2] + 0.5, PLATE_Z0 - 1, TRIM_Z + 1)
frame -= box(-15.4, 15.4, ESP_Y1 - 1, ESP_Y0 + 0.5, PLATE_Z1, TRIM_Z + 1)        # ESP32 volume (keeps backbone)
frame -= box(-14.5, 14.5, BAY[1], 62, PLATE_Z1, TRIM_Z + 1)                      # old camera-ear stubs by the pod

# ---------------- ADD: stack plate ----------------
plate = rounded_rect(52, 52, 6, PLATE_Z0, PLATE_Z1)
plate -= rounded_rect(21, 21, 4, PLATE_Z0 - 1, PLATE_Z1 + 1)                     # wiring / lightening window
for sx in (-1, 1):
    for sy in (-1, 1):
        plate -= cyl(sx * STACK_SP / 2, sy * STACK_SP / 2, M3 / 2, PLATE_Z0 - 1, PLATE_Z1 + 1)
frame += plate
for sx in (-1, 1):   # re-open stack holes in case backbone overlapped them
    for sy in (-1, 1):
        frame -= cyl(sx * STACK_SP / 2, sy * STACK_SP / 2, M3 / 2, PLATE_Z0 - 1, PLATE_Z1 + 1)
frame -= rounded_rect(21, 21, 4, PLATE_Z0 - 1, PLATE_Z1 + 1)

# ---------------- ADD: MTF-01 pod ----------------
hx, y0, y1 = POD_IN
pod = box(-hx - POD_WALL, hx + POD_WALL, y0 - POD_WALL, y1 + POD_WALL, PLATE_Z0, POD_TOP)
pod -= box(-hx, hx, y0, y1, PLATE_Z0 - 1, POD_TOP + 1)                           # open window + pocket
for (tx, ty) in MTF_HOLES:                                                        # corner tabs
    xs = (3.5, hx) if tx > 0 else (-hx, -3.5)
    ys = (y0, ty + 2.8) if ty < 40.95 else (ty - 2.8, y1)
    pod += box(xs[0], xs[1], ys[0], ys[1], PLATE_Z0, PLATE_Z0 + 3.0)
for (tx, ty) in MTF_HOLES:
    pod -= cyl(tx, ty, M2 / 2, PLATE_Z0 - 1, PLATE_Z0 + 4)
    pod -= cyl(tx, ty, 2.0, PLATE_Z0 - 1, PLATE_Z0 + 1.5)                          # screw-head counterbore
for zy in (y0 + 8, y1 - 8):                                                       # backup zip-tie slots
    pod -= box(-20, 20, zy - 1.6, zy + 1.6, PLATE_Z0 + 3.5, PLATE_Z0 + 5.0)
frame += pod

# ---------------- ADD: ESP32 cradle ----------------
cr = (box(ESP_RAIL[0], ESP_RAIL[1], ESP_Y1, ESP_Y0, PLATE_Z0, PLATE_Z1)
      + box(-ESP_RAIL[1], -ESP_RAIL[0], ESP_Y1, ESP_Y0, PLATE_Z0, PLATE_Z1))
for yb in (ESP_Y1, -40.0, -65.0):                                                 # crossbars (zip-tie anchors)
    cr += box(-ESP_RAIL[1], ESP_RAIL[1], yb, yb + 3.0 if yb != ESP_Y1 else yb + 3.5, PLATE_Z0, PLATE_Z1)
for sx in (-1, 1):
    for py in (ESP_Y0 - 3.5, ESP_Y1 + 3.5):
        cr += box(sx * 6 - 1.5, sx * 6 + 1.5, py - 1.5, py + 1.5, PLATE_Z0, ESP_POST_TOP)
frame += cr

# ---------------- ADD: truss prop guards ----------------
guards = []
for (mx, my) in MOTORS:
    g = ring(mx, my, G_IN, G_BOT_OUT, *G_BOT) + ring(mx, my, G_IN, G_TOP_OUT, *G_TOP)
    rb, rt = (G_IN + G_BOT_OUT) / 2, (G_IN + G_TOP_OUT) / 2
    for i in range(N_STRUT):
        a0 = 2 * math.pi * i / N_STRUT
        a1 = a0 + (math.pi / N_STRUT) * (1 if i % 2 == 0 else -1)
        p0 = (mx + rb * math.cos(a0), my + rb * math.sin(a0), G_BOT[0] + 1.2)
        p1 = (mx + rt * math.cos(a1), my + rt * math.sin(a1), G_TOP[0] + 1.0)
        g += strut(p0, p1, 0.95)
    # collar under the pad (outside M2 screw heads, below the bell) + 3 outboard spokes to the ring
    COLLAR = (6.9, 9.0, -1.2, 0.4)
    g += ring(mx, my, COLLAR[0], COLLAR[1], COLLAR[2], COLLAR[3])
    out = math.atan2(my, mx)
    for k in (-1, 0, 1):
        a = out + k * math.radians(60)
        rc = (COLLAR[0] + COLLAR[1]) / 2
        p0 = (mx + (rc - 0.6) * math.cos(a), my + (rc - 0.6) * math.sin(a), -0.4)
        p1 = (mx + rb * math.cos(a), my + rb * math.sin(a), G_BOT[0] + 1.25)
        g += strut(p0, p1, 1.2)
    g -= box(-BAY[0], BAY[0], -BAY[1], BAY[1], -10, 40)       # never intrude on the stack bay
    guards.append(g)
GUARDS = M.batch_boolean(guards, mf.OpType.Add)


def finish(man, label):
    # IMPROVE: flat seats clear of the bell, M2 clearance holes on the existing Ø9 circle
    for (mx, my) in MOTORS:
        man -= cyl(mx, my, SEAT_R, SEAT_Z0, TRIM_Z + 1)
    for (hx_, hy_) in holes:
        man -= cyl(hx_, hy_, M2 / 2, -7, 5, 24)
    parts = sorted(man.decompose(), key=lambda p: p.volume(), reverse=True)
    print(f"{label}: {len(parts)} components -> kept largest, dropped {sum(p.volume() for p in parts[1:]):.1f} mm3 of floating islands")
    return parts[0]


CORE = frame
frame_ng = finish(frame, "no guards")
frame = finish(frame + GUARDS, "with guards")

for (mx, my), g in zip(MOTORS, guards):
    collar = ring(mx, my, 6.9, 9.0, -1.2, 0.4)
    print(f"guard ({mx:+.0f},{my:+.0f}): collar fused to pad over {(collar ^ CORE).volume():5.1f} mm3; "
          f"ring anchored by 3 spokes + {len([j for j in ((g - collar) ^ CORE).decompose() if j.volume() > 0.5])} arm/backbone crossings")
RHO = 1.24
for name, man in (("with guards", frame), ("no guards", frame_ng)):
    print(f"{name:12s}: {man.volume()/1000:6.2f} cm3 -> {man.volume()/1000*RHO:5.1f} g PLA (solid upper bound)")

to_trimesh(frame).export(f"{OUT}/Synthara_Remix_Frame.stl")
to_trimesh(frame_ng).export(f"{OUT}/Synthara_Remix_Frame_NoGuards.stl")
print("EXPORTED")
