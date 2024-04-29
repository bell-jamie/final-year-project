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

# Upper Beam
UB_P1 = gmsh.model.geo.addPoint(0.0, H_T, 0.0, 0.0)
UB_P2 = gmsh.model.geo.addPoint(L, H_T, 0.0, 0.0)
UB_P3 = gmsh.model.geo.addPoint(L, H_T - H_UB, 0.0, 0.0)
UB_P4 = gmsh.model.geo.addPoint(0.0, H_T - H_UB, 0.0, 0.0)

UB_L1 = gmsh.model.geo.addLine(UB_P1, UB_P2)
UB_L2 = gmsh.model.geo.addLine(UB_P2, UB_P3)
UB_L3 = gmsh.model.geo.addLine(UB_P3, UB_P4)
UB_L4 = gmsh.model.geo.addLine(UB_P4, UB_P1)

gmsh.model.geo.mesh.setTransfiniteCurve(UB_L1, X + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(UB_L2, Y_UB + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(UB_L3, X + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(UB_L4, Y_UB + 1)

UB_CL1 = gmsh.model.geo.addCurveLoop([UB_L1, UB_L2, UB_L3, UB_L4])
UB_PS1 = gmsh.model.geo.addPlaneSurface([UB_CL1])
gmsh.model.geo.mesh.setTransfiniteSurface(UB_PS1)
gmsh.model.geo.mesh.setRecombine(2, UB_PS1)

# Interface
I_P1 = gmsh.model.geo.addPoint(0.0, H_T - H_UB, 0.0, 0.0)
I_P2 = gmsh.model.geo.addPoint(L, H_T - H_UB, 0.0, 0.0)
I_P3 = gmsh.model.geo.addPoint(L, H_LB, 0.0, 0.0)
I_P4 = gmsh.model.geo.addPoint(0.0, H_LB, 0.0, 0.0)

I_L1 = gmsh.model.geo.addLine(I_P1, I_P2)
I_L2 = gmsh.model.geo.addLine(I_P2, I_P3)
I_L3 = gmsh.model.geo.addLine(I_P3, I_P4)
I_L4 = gmsh.model.geo.addLine(I_P4, I_P1)

gmsh.model.geo.mesh.setTransfiniteCurve(I_L1, X + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(I_L2, Y_I + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(I_L3, X + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(I_L4, Y_I + 1)

I_CL1 = gmsh.model.geo.addCurveLoop([I_L1, I_L2, I_L3, I_L4])
I_PS1 = gmsh.model.geo.addPlaneSurface([I_CL1])
gmsh.model.geo.mesh.setTransfiniteSurface(I_PS1)
gmsh.model.geo.mesh.setRecombine(2, I_PS1)

# Lower Beam
LB_P1 = gmsh.model.geo.addPoint(0.0, H_LB, 0.0, 0.0)
LB_P2 = gmsh.model.geo.addPoint(L, H_LB, 0.0, 0.0)
LB_P3 = gmsh.model.geo.addPoint(L, 0.0, 0.0, 0.0)
LB_P4 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.0)

LB_L1 = gmsh.model.geo.addLine(LB_P1, LB_P2)
LB_L2 = gmsh.model.geo.addLine(LB_P2, LB_P3)
LB_L3 = gmsh.model.geo.addLine(LB_P3, LB_P4)
LB_L4 = gmsh.model.geo.addLine(LB_P4, LB_P1)

gmsh.model.geo.mesh.setTransfiniteCurve(LB_L1, X + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(LB_L2, Y_LB + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(LB_L3, X + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(LB_L4, Y_LB + 1)

LB_CL1 = gmsh.model.geo.addCurveLoop([LB_L1, LB_L2, LB_L3, LB_L4])
LB_PS1 = gmsh.model.geo.addPlaneSurface([LB_CL1])
gmsh.model.geo.mesh.setTransfiniteSurface(LB_PS1)
gmsh.model.geo.mesh.setRecombine(2, LB_PS1)

# Physical Groups
gmsh.model.addPhysicalGroup(2, [UB_PS1, I_PS1, LB_PS1], -1, "domain")

# Generate Mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.fltk.run()
gmsh.write(os.path.join(script_directory, "dcb.msh"))
gmsh.finalize()

# In order to use mesh with SfePy, run the following command in the terminal:
# sfepy-convert -d 2 dcb.msh dcb.vtk
