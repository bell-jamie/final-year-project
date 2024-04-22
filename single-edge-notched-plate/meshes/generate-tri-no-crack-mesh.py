# import pygmsh
import gmsh
import os

script_directory = os.path.dirname(os.path.abspath(__file__))

MESH_WIDTH = 1.0
MESH_HEIGHT = 1.0

COARSE_SIZE = 0.025
INTER_SIZE = 0.01
FINE_SIZE = 0.001

REFINE_HEIGHT = 0.05

gmsh.initialize()
gmsh.model.geo.addPoint(0.0, 0.0, 0.0, COARSE_SIZE, 1)
gmsh.model.geo.addPoint(MESH_WIDTH, 0.0, 0.0, COARSE_SIZE, 2)
gmsh.model.geo.addPoint(MESH_WIDTH, MESH_HEIGHT, 0.0, COARSE_SIZE, 3)
gmsh.model.geo.addPoint(0.0, MESH_HEIGHT, 0.0, COARSE_SIZE, 4)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.addPhysicalGroup(2, [1], 1, "Domain")
gmsh.model.geo.addPhysicalGroup(1, [1], 1, "Fixed")
gmsh.model.geo.addPhysicalGroup(1, [3], 2, "Load")

gmsh.model.mesh.field.add("Box", 11)
gmsh.model.mesh.field.setNumber(11, "VIn", FINE_SIZE)
gmsh.model.mesh.field.setNumber(11, "VOut", COARSE_SIZE)
gmsh.model.mesh.field.setNumber(11, "XMin", 0.0)
gmsh.model.mesh.field.setNumber(11, "XMax", MESH_WIDTH)
gmsh.model.mesh.field.setNumber(11, "YMin", MESH_HEIGHT / 2 - REFINE_HEIGHT / 2)
gmsh.model.mesh.field.setNumber(11, "YMax", MESH_HEIGHT / 2 + REFINE_HEIGHT / 2)
gmsh.model.mesh.field.setAsBackgroundMesh(11)

# gmsh.model.mesh.field.add("Box", 12)
# gmsh.model.mesh.field.setNumber(12, "VIn", INTER_SIZE)
# gmsh.model.mesh.field.setNumber(12, "VOut", COARSE_SIZE)
# gmsh.model.mesh.field.setNumber(12, "XMin", 0.0)
# gmsh.model.mesh.field.setNumber(12, "XMax", MESH_WIDTH / 2)
# gmsh.model.mesh.field.setNumber(12, "YMin", MESH_HEIGHT / 2 - REFINE_HEIGHT / 2)
# gmsh.model.mesh.field.setNumber(12, "YMax", MESH_HEIGHT / 2 + REFINE_HEIGHT / 2)
# gmsh.model.mesh.field.setAsBackgroundMesh(12)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.fltk.run()
gmsh.write(os.path.join(script_directory, "notchedPlateTriNoCrack.msh"))
gmsh.finalize()

# In order to use mesh with SfePy, run the following command in the terminal:
# sfepy-convert -d 2 notchedPlateTriNoCrack.msh notchedPlateTriNoCrack.vtk
