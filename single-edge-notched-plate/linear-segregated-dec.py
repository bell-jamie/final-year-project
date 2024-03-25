from __future__ import absolute_import
from datetime import datetime
from sfepy.mechanics import matcoefs, tensors

import numpy as np
import os


def strain(pb, displacement):
    # return pb.evaluate("ev_cauchy_strain.i.Omega(u)", mode="el_avg")
    pass
    # return problem.evaluate("ev_grad.i.Omega(u)", mode="qp")


def stress(strain):
    return strain * matcoefs.stiffness_from_youngpoisson(
        dim=2, young=E, poisson=NU, plane="strain"
    )


def stress_mod(strain, strain_in, damage_in):
    if np.trace(strain_in) >= 0:
        sigma = (damage_in**2 + ETA) * stress(
            strain
        )  # need to pass function to function
    else:
        sigma = tensors.get_deviator(
            damage_in**2 + ETA
        ) + tensors.get_volumetric_tensor(stress(strain))

    # I have an idea that I can generate filter matrices eg... [[0, 1, 0], [1, 1, 1], [0, 1, 0]
    # tensile filter = generate_filter_matrix(), compressive filter = ones() - tensile filter (inverted)
    # Then multiply ((damage^2 + eta) * stress tensor) and ((damage^2 + eta) * stress_dev + stress_vol) by filter matrix
    # Maybe just one filter... and then tensile = filter, compression = filter.T
    # The filter gets calculated in the step function and then passed to the stress_mod function
    # This way the only unknown is u_disp and calculations can be faster
    # Try and write this out by hand to validate the idea

    return sigma


def energy(displacement):
    epsilon = strain(displacement)
    if np.trace(epsilon) >= 0:
        phi = 0.5 * (strain * stress(epsilon))
    else:
        phi = (
            0.5
            * tensors.get_deviator(stress(epsilon))
            * tensors.get_volumetric_tensor(epsilon)
        )
    # return max(phi, phi_last)
    return phi


def init_phi(variables):
    phi = np.ones((variables["u_phase"].field.n_nod,), dtype=np.float64)
    variables.set_state_parts({"phi": phi})


def step_hook(pb, ts, variables):
    # u_disp = variables["displacement"]
    strain = pb.evaluate("ev_cauchy_strain.i.Omega(u_disp)", mode="qp")

    print(strain)
    print(strain.shape)

    # Strain is formatted as [xx, yy, 2xy] to exploit symmetry of [[xx, xy], [yx, yy]]
    # By default all the tensors use the symmetric tensor format
    # By including a flag this can be changed: tensors.get_volumetric_tensor(epsilon, symmetric=False)]

    """

    phi = np.zeros(strain.shape[0])  # phi value for each element

    strain_2D = np.zeros((strain.shape[0], 2, 2))

    for i in range(strain_2D.shape[0]):
        strain_2D[i, 0, 0] = strain[i, 0]
        strain_2D[i, 1, 1] = strain[i, 1]
        strain_2D[i, 0, 1] = strain[i, 2] / 2
        strain_2D[i, 1, 0] = strain[i, 2] / 2

    for i in range(phi.shape[0]):
        phi[i] = (
            0.5 * (strain_2D[i] * stress(strain_2D[i]))
            if np.trace(strain_2D[i]) >= 0
            else 0.5
            * (
                tensors.get_deviator(stress(strain_2D[i]))
                * tensors.get_deviator(strain_2D[i])
            )
        )

    for i in range(strain.shape[0]):
        for j in range(strain.shape[1]):
            # Psuedo-trace(strain) as strain is only given as xx, yy, zz
            vector_sum = np.sum(strain[i, j, :, :])

            phi[i, j] = (
                0.5 * (strain * stress(strain))
                if vector_sum >= 0
                else 0.5
                * (tensors.get_deviator(stress(strain)) * tensors.get_deviator(strain))
            )
    """

    elems = strain.shape[0]
    phi = np.zeros(elems)
    # If phi is not in variables, set phi_old to phi
    phi_old = variables["phi"] if "phi" in variables else phi

    for i in range(elems):
        strain_local = strain[i][0]
        print(strain_local)
        phi[i] = max(
            (
                0.5 * (strain_local * stress(strain_local))
                if tensors.get_trace(strain_local).all() >= 0
                else 0.5
                * (
                    tensors.get_deviator(stress(strain_local))
                    * tensors.get_deviator(strain_local)
                )
            ),
            phi_old[i],
        )

    variables["phi"] = phi

    # variables["phi"] = max(variables["phi"], phi_new)


def post_process(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    from sfepy.base.base import Struct
    from sfepy.mechanics.tensors import get_von_mises_stress

    ev = pb.evaluate
    stress = ev("ev_cauchy_stress.i.Omega(m.C, u_disp)", mode="el_avg")

    vms = get_von_mises_stress(stress.squeeze())
    vms.shape = (vms.shape[0], 1, 1, 1)
    out["von_mises_stress"] = Struct(
        name="output_data", mode="cell", data=vms, dofs=None
    )

    pb.save_state(os.path.join(pb.output_dir, "test.vtk"), out=out)

    return out  # needed?


# Constants
T0 = 0.0
T1 = 1.0

E = 210e3
NU = 0.3

LS = 0.0075
GC = 2.7
ETA = 1e-15

ORDER = 2
DEGREE = 2 * ORDER

####### TEMPORARY MESH #######
from sfepy import data_dir
from sfepy.discrete.fem import Mesh

filename_mesh = data_dir + "/meshes/2d/rectangle_tri.mesh"
##############################

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_directory = os.path.dirname(__file__)
# filename_mesh = os.path.join(script_directory, "meshes", "notchedPlateTriangular.vtk")
save_directory = os.path.join(
    script_directory,
    "files",
    os.path.splitext(os.path.basename(__file__))[0] + "-py",
    current_datetime,
)

os.makedirs(save_directory, exist_ok=True)

options = {
    "nls": "newton",
    "ls": "ls",
    "step_hook": "step_hook",
    #'parametric_hook' : 'parametric_hook', # can be used to programatically change problem
    "output_dir": save_directory,  # this might not work the way I thought it does
    "post_process_hook": "post_process",
}

regions = {
    "Omega": "all",
    "Load": ("vertices in (y > 8.9)", "facet"),  # should be y > 0.49
    "Fixed": ("vertices in (y < -8.9)", "facet"),  # should be y < -0.49
}

fields = {
    "displacement": ("real", 2, "Omega", ORDER, "H1"),
    "damage": ("real", 1, "Omega", ORDER, "H1"),
}

variables = {
    "u_disp": ("unknown field", "displacement", 0),
    "v_disp": ("test field", "displacement", "u_disp"),
    "u_phase": (
        "unknown field",
        "damage",
        1,
    ),
    "v_phase": ("test field", "damage", "u_phase"),
    "phi": ("parameter field", "displacement", {"setter": "init_phi"}),
}

# may not be necessary
materials = {
    "m": (
        {
            "C": matcoefs.stiffness_from_youngpoisson(
                dim=2, young=E, poisson=NU, plane="strain"
            ),
            "GCLS": GC * LS,
            "GC_LS": GC / LS,
        },
    ),
}

integrals = {
    # "i": ORDER,
    "i": DEGREE,
}

# sfepy/examples/acoustics/acoustics3d.py - has some interesting equation syntax
# should we solve damage first? - jamie for tomorrow you were about to try and make the phi energy function work with a step hook...
equations = {
    "eq_disp": """dw_lin_elastic.i.Omega(m.C, v_disp, u_disp) = 0""",
    # "eq_phase": """dw_laplace.i.Omega(m.GCLS, v_phase, u_phase) + 2 * phi * dw_dot.i.Omega(v_phase, u_phase) + dw_dot.i.Omega(m.GC_LS, v_phase, u_phase) = dw_integrate.i.Omega(m.GC_LS, v_phase)""",
    "eq_phase": """dw_laplace.i.Omega(m.GCLS, v_phase, u_phase) + dw_dot.i.Omega(m.GC_LS, v_phase, u_phase) = dw_integrate.i.Omega(m.GC_LS, v_phase)""",
}

functions = {
    "strain": (strain,),
    "stress": (stress,),
    "stress_mod": (stress_mod,),
    "init_phi": (init_phi,),
    "energy": (energy,),
}

ics = {
    "phase": ("Omega", {"damage": 1.0}),
    "disp": ("Omega", {"displacement": 0.0}),
}

ebcs = {
    "fixed": ("Fixed", {"u_disp.all": 0.0}),
    "load": ("Load", {"u_disp.1": 1.0}),  # maybe set as a function of time
}

solvers = {
    "ls": ("ls.scipy_direct", {}),
    "nls": (
        "nls.newton",
        {
            "i_max": 100,
            "eps_a": 1e-6,
        },
    ),
    "ts": (
        "ts.simple",
        {
            "t0": T0,
            "t1": T1,
            "dt": 0.01,  # need to make this dynamic...
            #'dt'     : None,
            #'n_step' : 5,
            "quasistatic": True,
            "verbose": 1,
        },
    ),
}
