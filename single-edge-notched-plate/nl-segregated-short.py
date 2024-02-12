from sfepy import data_dir
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson

E = 126e9
nu = 0.3

filename_mesh = data_dir + '/notchedPlateTriangular.mesh'

regions = {
    'omega' : 'all',
    'top' : '', # find all the vertices on the top of the plate (x >= 0.5)
    'bottom' : '', # find all the vertices on the bottom of the plate (x <= 0.5)
}

materials = {
    'solid' : ({'D' : stiffness_from_youngpoisson(E, nu)}),
}

fields = {
    'displacement' : ('real', 'vector', 'Omega', 1),
    'phase-field' : ('real', 'scalar', 'Omega', 1),
}

equations = {
    'balance_of_forces' : """dw_lin_elastic_iso.i.Omega( Solid.D, v, u ) = 0""",
    'damage_evolution' : """dw_laplace.i.Omega( Solid.D, v, u ) + dw_volume_dot.i.Omega( f, v, u ) = 0""", # revisit this
}

variables = {
    'u_pf' : ('unknown field', 'phase-field', 0), # is damage being solved first? 0 -> 1
    'v_pf' : ('test field', 'phase-field', 'u_pf'),
    'u' : ('unknown field', 'displacement', 1),
    'v' : ('test field', 'displacement', 'u'),
}

ebcs = {
    'fixed' : ('Bottom', {'u.0' : 0.0, 'u.1' : 0.0}), # fix x and y of the bottom vertices
}

solvers = {
    'ls' : ('ls.scipy_direct', {}),
    'newton' : ('nls.newton', {
        'i_max' : 1, # maximum number of iterations - make this equal to Julia code
        'eps_a' : 1e-6, # res norm tolerance
    }),
}