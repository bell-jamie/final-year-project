import Gmsh: gmsh

w = 1
h = 1

x = 50
y = 50
pow = 50

gmsh.initialize()
gmsh.model.add("notchedPlateTransfinite")

# Corner Vertices
gmsh.model.geo.addPoint(-w / 2, h / 2, 0, 1)
gmsh.model.geo.addPoint(w / 2, h / 2, 0, 2)
gmsh.model.geo.addPoint(w / 2, -h / 2, 0, 3)
gmsh.model.geo.addPoint(-w / 2, -h / 2, 0, 4)

# Crack Vertices
#gmsh.model.geo.addPoint(-w / 2, 0, 0, 5)
#gmsh.model.geo.addPoint(0, 0, 0, 6)

# Lines
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

#gmsh.model.geo.addLine(5, 6, 5) # crack line

# Transfinite Lines
gmsh.model.geo.mesh.setTransfiniteCurve(1, x, "Progression", 1)
gmsh.model.geo.mesh.setTransfiniteCurve(2, y, "Bump", pow)
gmsh.model.geo.mesh.setTransfiniteCurve(3, x, "Progression", 1)
gmsh.model.geo.mesh.setTransfiniteCurve(4, y, "Bump", pow)

# Curve Loop and Surface
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.mesh.setTransfiniteSurface(1)
gmsh.model.geo.mesh.setRecombine(2, 1)

#gmsh.model.geo.addPhysicalGroup(0, [1, 2], 1, "load") # add top points to physical group
#gmsh.model.geo.addPhysicalGroup(1, [1], 1, "load") # add top line to physical group
#gmsh.model.geo.addPhysicalGroup(0, [3, 4], 2, "fixed") # add bottom points to physical group
#gmsh.model.geo.addPhysicalGroup(1, [3], 2, "fixed") # add bottom line to physical group
#gmsh.model.geo.addPhysicalGroup(1, [5], 3, "crack") # add crack line to physical group
gmsh.model.geo.addPhysicalGroup(2, [1], 1, "domain")

# Generate Mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.fltk.run()
gmsh.write(joinpath(@__DIR__, "notchedPlateTransfinite.msh"))
gmsh.finalize()

# In order to use mesh with SfePy, run the following command in the terminal:
# sfepy-convert -d 2 notchedPlateTransfinite.msh notchedPlateTransfinite.vtk
