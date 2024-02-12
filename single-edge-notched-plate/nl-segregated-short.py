from sfepy import data_dir
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson

E = 126e9
nu = 0.3

filename_mesh = data_dir + '/notchedPlateTriangular.mesh'

regions = {
    'Omega' : 'all',
    'Top' : '', # find all the vertices on the top of the plate (x >= 0.5)
    'Bottom' : '', # find all the vertices on the bottom of the plate (x <= 0.5)
}

materials = {
    'Solid' : ({'D' : stiffness_from_youngpoisson(E, nu)}),
}

fields = {
    'Displacement' : ('real', 'vector', 'Omega', 1),
    'Damage' : ('real', 'scalar', 'Omega', 1),
}

equations = {
    'balance_of_forces' : """dw_lin_elastic_iso.i.Omega( Solid.D, v, u ) = 0""",
    'damage_evolution' : """dw_laplace.i.Omega( Solid.D, v, u ) + dw_volume_dot.i.Omega( f, v, u ) = 0""", # revisit this
}

variables = {
    
}