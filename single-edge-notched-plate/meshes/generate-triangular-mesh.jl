import Gmsh: gmsh

cw = 1e-3
w = 1
h = 1

ms_b = 1e-1   # mesh size for bulk
ms_c = 8e-2     # mesh size for crack
ms_bc = 1e-4   # mesh size for crack in bulk

gmsh.initialize()
gmsh.model.add("notchedPlateTriangular")

# Corner Vertices
gmsh.model.geo.addPoint(-w / 2, h / 2, 0, ms_b, 1)
gmsh.model.geo.addPoint(w / 2, h / 2, 0, ms_b, 2)
gmsh.model.geo.addPoint(w / 2, -h / 2, 0, ms_b, 3)
gmsh.model.geo.addPoint(-w / 2, -h / 2, 0, ms_b, 4)

# Crack Vertices
gmsh.model.geo.addPoint(-w / 2, -cw / 2, 0, ms_c, 5)
gmsh.model.geo.addPoint(0, -cw / 2, 0, ms_c, 6)
gmsh.model.geo.addPoint(0, cw / 2, 0, ms_c, 7)
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
gmsh.model.geo.addPlaneSurface([1], 1)

#gmsh.model.geo.addPhysicalGroup(0, [1, 2], 1, "load") # add top points to physical group
gmsh.model.geo.addPhysicalGroup(1, [1], 1, "load") # add top line to physical group
#gmsh.model.geo.addPhysicalGroup(0, [3, 4], 2, "fixed") # add bottom points to physical group
gmsh.model.geo.addPhysicalGroup(1, [3], 2, "fixed") # add bottom line to physical group
gmsh.model.geo.addPhysicalGroup(2, [1], 1, "domain")

# Refine Crack Region
gmsh.model.mesh.field.add("Box", 11)
gmsh.model.mesh.field.setNumber(11, "VIn", ms_bc)
gmsh.model.mesh.field.setNumber(11, "VOut", ms_c)
gmsh.model.mesh.field.setNumber(11, "XMin", 0 - 3 * cw)
gmsh.model.mesh.field.setNumber(11, "XMax", w / 2)
gmsh.model.mesh.field.setNumber(11, "YMin", -5 * cw)
gmsh.model.mesh.field.setNumber(11, "YMax", 5 * cw)
gmsh.model.mesh.field.setAsBackgroundMesh(11)

# Generate Mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
#gmsh.fltk.run()
gmsh.write(joinpath(@__DIR__, "notchedPlateTriangular.msh"))
gmsh.write(joinpath(@__DIR__, "notchedPlateTriangular.vtk"))
gmsh.finalize()