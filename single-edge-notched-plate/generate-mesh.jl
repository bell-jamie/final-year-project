using gmsh_jll
include(gmsh_jll.gmsh_api)

cw = 1e-5
w = 1
h = 1

ms_b = 1 / 8    # mesh size for bulk
ms_c = 1e-2     # mesh size for crack

gmsh.initialize()
gmsh.model.add("notchedPlateTriangular")

# Corner Vertices
gmsh.model.geo.addPoint(-w / 2, h / 2, 0, ms_b, 1)
gmsh.model.geo.addPoint(w / 2, h / 2, 0, ms_b, 2)
gmsh.model.geo.addPoint(w / 2, -h / 2, 0, ms_b, 4)
gmsh.model.geo.addPoint(-w / 2, -h / 2, 0, ms_b, 5)

# Midpoint Vertex
gmsh.model.geo.addPoint(w / 2, 0, 0, ms_c, 3)

# Crack Vertices
gmsh.model.geo.addPoint(-w / 2, 0 - cw / 2, 0, ms_c, 6)
gmsh.model.geo.addPoint(0, 0, 0, ms_c, 7)
gmsh.model.geo.addPoint(-w / 2, 0 + cw / 2, 0, ms_c, 8)

# Lines
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 5, 4)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 1, 8)

# Curve Loop and Surface
gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
s = gmsh.model.geo.addPlaneSurface([1], 1)

#gmsh.model.geo.addPhysicalGroup(0, [1, 2], 1, "top") # add top points to physical group
gmsh.model.geo.addPhysicalGroup(1, [1], 1, "top") # add top line to physical group

#gmsh.model.geo.addPhysicalGroup(0, [4, 5], 2, "bottom") # add bottom points to physical group
gmsh.model.geo.addPhysicalGroup(1, [4], 2, "bottom") # add bottom line to physical group

# Generate Mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
#gmsh.fltk.run()

gmsh.write("notchedPlateTriangular.msh")
gmsh.finalize()