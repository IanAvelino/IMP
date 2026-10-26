import math, numpy as np, trimesh, manifold3d as mf
M=mf.Manifold
D="/Users/ianavelino/Desktop/Claude/Drone_CAD/Synthara_Remix"
def load(p):
    t=trimesh.load(p); return t, M(mf.Mesh(vert_properties=np.asarray(t.vertices,np.float32),tri_verts=np.asarray(t.faces,np.uint32)))
def box(x0,x1,y0,y1,z0,z1): return M.cube((x1-x0,y1-y0,z1-z0)).translate((x0,y0,z0))
def cyl(x,y,r,z0,z1): return M.cylinder(z1-z0,r,circular_segments=96).translate((x,y,z0))
MOT=[(sx*60.0,sy*49.0) for sx in(-1,1) for sy in(-1,1)]
env={
 'SpeedyBee stack (ESC 45.6x44, z -2.5..20)': box(-22.8,22.8,-22,22,-2.5,20),
 'LANNRC bells (Ø19.3, z 3.05..17.8)': M.batch_boolean([cyl(x,y,9.65,3.05,17.8) for x,y in MOT],mf.OpType.Add),
 '3in prop disks (+0.5mm, z 17.8..24)': M.batch_boolean([cyl(x,y,38.6,17.8,24.0) for x,y in MOT],mf.OpType.Add),
 'MTF-01 body (29.3x17, face at z -3)': box(-8.5,8.5,26.3,55.6,-2.99,11.5),
 'ESP32 PCB+module (z 6..13)': box(-14,14,-80.5,-25.5,6.01,13),
 'ESP32 header pins (to z -2.5)': box(-14,-11.4,-80.5,-25.5,-2.5,6)+box(11.4,14,-80.5,-25.5,-2.5,6),
 'Top deck 42x54 (z 21.5..24.5)': box(-21,21,-27,27,21.5,24.5),
}
for f in ('Synthara_Remix_Frame.stl','Synthara_Remix_Frame_NoGuards.stl'):
    t,m=load(f"{D}/{f}")
    print(f"\n== {f}: watertight={t.is_watertight}, bodies={len(m.decompose())}, bbox={np.round(t.bounds,1).tolist()}")
    for k,e in env.items():
        v=(m^e).volume()
        print(f"   {'OK ' if v<0.5 else 'HIT'}  {k:42s} overlap {v:7.2f} mm3")
    # motor seat flatness + hole size check via slice
    for z in (1.5,):
        smalls=[]
        for poly in m.slice(z).to_polygons():
            P=np.asarray(poly); x,y=P[:,0],P[:,1]
            A=abs(0.5*np.sum(x*np.roll(y,-1)-np.roll(x,-1)*y))
            c=P.mean(0)
            if A<6 and min(math.hypot(c[0]-a,c[1]-b) for a,b in MOT)<6: smalls.append(2*math.sqrt(A/math.pi))
        print(f"   motor bolt holes at z={z}: {len(smalls)} found, dia {min(smalls):.2f}-{max(smalls):.2f} mm")
    # arms still reach each pad: motor pad must be connected (single body already) and have material at z 1.5 within r 4-8
    for x,y in MOT:
        pad=(m^cyl(x,y,8,0.2,2.8)).volume()
        assert pad>50, f"pad missing at {x},{y}"
    print("   all four motor pads present")
