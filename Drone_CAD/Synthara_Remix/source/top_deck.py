import cadquery as cq
D="/Users/ianavelino/Desktop/Claude/Drone_CAD/Synthara_Remix"
W,L,T=42.0,54.0,2.5          # corners stay 1.8mm+ clear of the prop-guard top band
deck=cq.Workplane("XY").rect(W,L).extrude(T).edges("|Z").fillet(5)
cuts=cq.Workplane("XY").pushPoints([(sx*15.25,sy*15.25) for sx in(-1,1) for sy in(-1,1)]).circle(1.6).extrude(T)   # M3 stack holes
cuts=cuts.union(cq.Workplane("XY").pushPoints([(-17.8,0),(17.8,0)]).slot2D(20.0,3.2,90).extrude(T))              # battery strap slots
cuts=cuts.union(cq.Workplane("XY").center(0,-21.5).slot2D(12,5,0).extrude(T))                                   # XT30 lead pass-through (rear)
cuts=cuts.union(cq.Workplane("XY").polyline([(0,24.5),(-5,17.5),(5,17.5)]).close().extrude(T))                  # FRONT arrow (+Y)
cuts=cuts.union(cq.Workplane("XY").pushPoints([(0,5),(0,-8)]).slot2D(18,7,0).extrude(T))                        # lightening
deck=deck.cut(cuts)
v=deck.val().Volume()
print(f"top deck: {v/1000:.2f} cm3 -> {v/1000*1.24:.1f} g PLA")
cq.exporters.export(deck,f"{D}/Synthara_Remix_TopDeck.step")
cq.exporters.export(deck,f"{D}/Synthara_Remix_TopDeck.stl")
