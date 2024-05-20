import gmsh
import os

script_directory = os.path.dirname(os.path.abspath(__file__))

WIDTH = 1.0
HEIGHT = 1.0

LS = 0.0075
COARSE = 5 * LS
FINE = LS / 10
ULTRAFINE = LS / 50
CRACK_HEIGHT = HEIGHT / 1000

gmsh.initialize()

P1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, COARSE)
P2 = gmsh.model.geo.addPoint(WIDTH, 0.0, 0.0, COARSE)
P3 = gmsh.model.geo.addPoint(WIDTH, HEIGHT, 0.0, COARSE)
P4 = gmsh.model.geo.addPoint(0.0, HEIGHT, 0.0, COARSE)
P5 = gmsh.model.geo.addPoint(0.0, HEIGHT / 2 + CRACK_HEIGHT, 0.0, COARSE)
P6 = gmsh.model.geo.addPoint(WIDTH / 2, HEIGHT / 2 + CRACK_HEIGHT, 0.0, ULTRAFINE)
P7 = gmsh.model.geo.addPoint(WIDTH / 2, HEIGHT / 2 - CRACK_HEIGHT, (0.0), ULTRAFINE)
P8 = gmsh.model.geo.addPoint(0.0, HEIGHT / 2 - CRACK_HEIGHT, 0.0, COARSE)

L1 = gmsh.model.geo.addLine(P1, P2)
L2 = gmsh.model.geo.addLine(P2, P3)
L3 = gmsh.model.geo.addLine(P3, P4)
L4 = gmsh.model.geo.addLine(P4, P5)
L5 = gmsh.model.geo.addLine(P5, P6)
L6 = gmsh.model.geo.addLine(P6, P7)
L7 = gmsh.model.geo.addLine(P7, P8)
L8 = gmsh.model.geo.addLine(P8, P1)

CL1 = gmsh.model.geo.addCurveLoop([L1, L2, L3, L4, L5, L6, L7, L8])
PS1 = gmsh.model.geo.addPlaneSurface([CL1])

gmsh.model.geo.addPhysicalGroup(2, [PS1], -1, "Domain")
gmsh.model.geo.addPhysicalGroup(1, [L1], -1, "Fixed")
gmsh.model.geo.addPhysicalGroup(1, [L3], -1, "Load")

# gmsh 4.13.0 manual - page 38, field mesh

gmsh.model.mesh.field.add("Box", 11)
gmsh.model.mesh.field.setNumber(11, "VIn", FINE)
gmsh.model.mesh.field.setNumber(11, "VOut", COARSE)
gmsh.model.mesh.field.setNumber(11, "XMin", 0.95 * WIDTH / 2)
gmsh.model.mesh.field.setNumber(11, "XMax", WIDTH)
gmsh.model.mesh.field.setNumber(11, "YMin", HEIGHT / 2 - COARSE)
gmsh.model.mesh.field.setNumber(11, "YMax", HEIGHT / 2 + COARSE)
gmsh.model.mesh.field.setAsBackgroundMesh(11)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.fltk.run()
gmsh.write(os.path.join(script_directory, "senp-geometric-crack.msh"))
gmsh.finalize()

# In order to use mesh with SfePy, run the following command in the terminal:
# sfepy-convert -d 2 senp-geometric-crack.msh senp-geometric-crack.vtk
