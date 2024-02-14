from sfepy import data_dir
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson

def applied_displacement(ts, co, **kwargs):
    return 0.02 * ts.dt # temporary function

E = 126e9
nu = 0.3

filename_mesh = data_dir + '/notchedPlateTriangular.mesh'

regions = {
    'omega' : 'all',
    'top' : ('vertices in (y > 0.49)', 'facet'),
    'bottom' : ('vertices in (y < -0.49)', 'facet'),
}

materials = {
    'solid' : ({'D' : stiffness_from_youngpoisson(E, nu)}),
}

fields = {
    'displacement' : ('real', 'vector', 'Omega', 2), # 2nd order approximation
    'phase-field' : ('real', 'scalar', 'Omega', 2),
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
    'fixed' : ('Bottom', {'u.all' : 0.0}), # fix x and y of the bottom vertices
    'load' : ('Top', {'u.0' : 0.0, 'u.1' : 'applied_displacement'})
}

ics = {
    'u_ic' : ('Omega', {'u.all' : 0.0}), # initial condition for displacement
    's_ic' : ('Omega', {'u_pf' : 1.0}), # initial condition for phase-field
}

functions = {
    'applied_displacement' : (applied_displacement,),
}

solvers = {
    'ls' : ('ls.scipy_direct', {}),
    'newton' : ('nls.newton', {
        'i_max' : 5, # maximum number of iterations
        'eps_a' : 1e-6, # residual norm tolerance
    }),
}