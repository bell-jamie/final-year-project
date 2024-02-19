from argparse import ArgumentParser
import numpy as np

import sys
sys.path.append('.')

from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Function, Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.mechanics.matcoefs import stiffness_from_lame

def lame_parameters(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) # Plane strain
    lam = 2.0 * mu * lam / (lam + 2.0 * mu) # Plane stress
    return lam, mu

def main():
    from sfepy import data_dir

    parser = ArgumentParser()
    parser.add_argument('--version', action = 'version', version = '%(prog)s')
    options = parser.parse_args()

    from sfepy.mesh.mesh_generators import gen_block_mesh

    # for generating the mesh, install MESHIO to go from .msh -> .vtk
    # already installed as sfepy dependency
    # meshio convert    input.msh output.vtk 

    length = (100) * 1e-3 # beam length in (mm) m
    thickness = (5) * 1e-3 # beam thickness in (mm) m
    element_length = (0.5) * 1e-3 # element length in (mm) m

    E = 70.0e9 # Young's modulus
    nu = 0.33 # Poisson's ratio

    order = 2 # order of element interpolation

    mesh = gen_block_mesh((length, thickness), (length / element_length, thickness / element_length), (length / 2, thickness / 2), mat_id = 0, name = 'beam', coors = None, verbose = True)
    #mesh = Mesh.from_file(data_dir + '/meshes/2d/rectangle_tri.mesh')
    domain = FEDomain('domain', mesh)

    min_x, max_x = domain.get_mesh_bounding_box()[:,0] # bounding box for selecting edges
    eps = 1e-8 * (max_x - min_x)
    omega = domain.create_region('Omega', 'all') # whole domain
    gamma1 = domain.create_region('Gamma1',
                                  'vertices in x < %.10f' % (min_x + eps),
                                  'facet') # gamma 1 region
    gamma2 = domain.create_region('Gamma2',
                                  'vertices in x > %.10f' % (max_x - eps),
                                  'facet') # gamma 2 region
    
    field = Field.from_args('fu', np.float64, 'vector', omega,
                            approx_order = order)
    
    u = FieldVariable('u', 'unknown', field) # unknown variable
    v = FieldVariable('v', 'test', field, primary_var_name = 'u') # test variable

    lam, mu = lame_parameters(E, nu)

    m = Material('m', D = stiffness_from_lame(dim = 2, lam = lam, mu = mu)) # linear elastic material
    f = Material('f', val = [[0.0], [0.0]]) # volume force constant column vector
    force = Material('force', val = [[0.0], [-5e3/thickness]]) # distributed force vector

    integral = Integral('i', order = 2 * order) # numerical quadrature for integration

    t1 = Term.new('dw_lin_elastic(m.D, v, u)',
                  integral, omega, m = m, v = v, u = u)
    t2 = Term.new('dw_volume_lvf(f.val, v)', integral, omega, f = f, v = v)
    t3 = Term.new('dw_surface_ltr(force.val, v)', integral, gamma2, force = force, v = v)
    eq1 = Equation('balance', t1 + t2 - t3)
    #eq2 = Equation('distributed_force', t3)
    eqs = Equations([eq1])

    fix_u = EssentialBC('fix_u', gamma1, {'u.all' : 0.0}) # clamp left edge

    ls = ScipyDirect({}) # define solvers, problem is linear and should converge in one iteration

    nls_status = IndexedStruct()
    nls = Newton({}, lin_solver = ls, status = nls_status)

    pb = Problem('cantilever', equations = eqs) # create problem instance
    pb.save_regions_as_groups('regions') # save regions as vtk file

    pb.set_bcs(ebcs = Conditions([fix_u])) # set the boundary conditions
    #pd.set_bcs()

    pb.set_solver(nls) # solve the problem—when a nonlinear solver is selected, default ts.stationary is created automatically

    status = IndexedStruct()
    variables = pb.solve(status = status)

    print('Nonlinear solver status:\n', nls_status)
    print('Stationary solver status:\n', status)

    pb.save_state('cantilever.vtk', variables) # save solution as vtk

    # sfepy-view cantilever.vtk -2
    # sfepy-view cantilever.vtk -2 -f u:wu:p0 1:vw:p0 # show displacements by shifting mesh

if __name__ == '__main__':
    main()
