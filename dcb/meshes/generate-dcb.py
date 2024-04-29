import gmsh
import os

script_directory = os.path.dirname(os.path.abspath(__file__))

# Constants
L = 100e-3
H_UB = 1.33e-3
H_LB = 2.66e-3
H_I = 0.1e-3
H_T = H_UB + H_I + H_LB

LS = 0.03e-3
GC_I = 281
GC_B = 10 * GC_I

X = 1200
Y_UB = 6
Y_I = 6
Y_LB = 12

gmsh.initialize()

P1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
P2 = gmsh.model.geo.addPoint(L, 0.0, 0.0)
P3 = gmsh.model.geo.addPoint(L, H_LB, 0.0)
P4 = gmsh.model.geo.addPoint(L, H_LB + H_I, 0.0)
P5 = gmsh.model.geo.addPoint(L, H_T, 0.0)
P6 = gmsh.model.geo.addPoint(0.0, H_T, 0.0)
P7 = gmsh.model.geo.addPoint(0.0, H_LB + H_I, 0.0)
P8 = gmsh.model.geo.addPoint(0.0, H_LB, 0.0)

L1 = gmsh.model.geo.addLine(P1, P2)
L2 = gmsh.model.geo.addLine(P2, P3)
L3 = gmsh.model.geo.addLine(P3, P4)
L4 = gmsh.model.geo.addLine(P4, P5)
L5 = gmsh.model.geo.addLine(P5, P6)
L6 = gmsh.model.geo.addLine(P6, P7)
L7 = gmsh.model.geo.addLine(P7, P8)
L8 = gmsh.model.geo.addLine(P8, P1)

gmsh.model.geo.mesh.setTransfiniteCurve(L1, X)
gmsh.model.geo.mesh.setTransfiniteCurve(L2, Y_LB)
gmsh.model.geo.mesh.setTransfiniteCurve(L3, Y_I)
gmsh.model.geo.mesh.setTransfiniteCurve(L4, Y_UB)
gmsh.model.geo.mesh.setTransfiniteCurve(L5, X)
gmsh.model.geo.mesh.setTransfiniteCurve(L6, Y_UB)
gmsh.model.geo.mesh.setTransfiniteCurve(L7, Y_I)
gmsh.model.geo.mesh.setTransfiniteCurve(L8, Y_LB)

CL1 = gmsh.model.geo.addCurveLoop([L1, L2, L3, L4, L5, L6, L7, L8])
PS1 = gmsh.model.geo.addPlaneSurface([CL1])
gmsh.model.geo.mesh.setTransfiniteSurface(PS1, "Left", [P1, P2, P5, P6])
gmsh.model.geo.mesh.setRecombine(2, PS1)

# Physical Groups
gmsh.model.addPhysicalGroup(2, [PS1], -1, "domain")

# Generate Mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.fltk.run()
gmsh.write(os.path.join(script_directory, "dcb.msh"))
gmsh.finalize()

# In order to use mesh with SfePy, run the following command in the terminal:
# sfepy-convert -d 2 dcb.msh dcb.vtk
