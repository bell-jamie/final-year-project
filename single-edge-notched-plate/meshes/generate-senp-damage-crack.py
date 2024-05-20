import gmsh
import os

script_directory = os.path.dirname(os.path.abspath(__file__))

MESH_WIDTH = 1.0
MESH_HEIGHT = 1.0

LS = 0.0075
COARSE_SIZE = 2 * LS
FINE_SIZE = LS / 5
REFINE_HEIGHT = 0.05

gmsh.initialize()

P1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
P2 = gmsh.model.geo.addPoint(MESH_WIDTH, 0.0, 0.0)
P3 = gmsh.model.geo.addPoint(MESH_WIDTH, MESH_HEIGHT, 0.0)
P4 = gmsh.model.geo.addPoint(0.0, MESH_HEIGHT, 0.0)

L1 = gmsh.model.geo.addLine(P1, P2)
L2 = gmsh.model.geo.addLine(P2, P3)
L3 = gmsh.model.geo.addLine(P3, P4)
L4 = gmsh.model.geo.addLine(P4, P1)

CL1 = gmsh.model.geo.addCurveLoop([L1, L2, L3, L4])
PS1 = gmsh.model.geo.addPlaneSurface([CL1])

gmsh.model.geo.addPhysicalGroup(2, [PS1], -1, "Domain")
gmsh.model.geo.addPhysicalGroup(1, [L1], -1, "Fixed")
gmsh.model.geo.addPhysicalGroup(1, [L3], -1, "Load")

gmsh.model.mesh.field.add("Box", 11)
gmsh.model.mesh.field.setNumber(11, "VIn", FINE_SIZE)
gmsh.model.mesh.field.setNumber(11, "VOut", COARSE_SIZE)
gmsh.model.mesh.field.setNumber(11, "XMin", 0.0)
gmsh.model.mesh.field.setNumber(11, "XMax", MESH_WIDTH)
gmsh.model.mesh.field.setNumber(11, "YMin", MESH_HEIGHT / 2 - REFINE_HEIGHT / 2)
gmsh.model.mesh.field.setNumber(11, "YMax", MESH_HEIGHT / 2 + REFINE_HEIGHT / 2)
gmsh.model.mesh.field.setAsBackgroundMesh(11)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.fltk.run()
gmsh.write(os.path.join(script_directory, "senp-damage-crack.msh"))
gmsh.finalize()

# In order to use mesh with SfePy, run the following command in the terminal:
# sfepy-convert -d 2 senp-damage-crack.msh senp-damage-crack.vtk
