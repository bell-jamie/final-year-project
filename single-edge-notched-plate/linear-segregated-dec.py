from __future__ import absolute_import
from datetime import datetime
from sfepy.mechanics import matcoefs, tensors

# from sfepy.discrete import Problem
from sfepy.discrete.fem import Mesh
from sfepy.discrete.conditions import EssentialBC, Conditions
from sfepy.discrete import Problem

import numpy as np
import os


def strain(pb, displacement):
    # return pb.evaluate("ev_cauchy_strain.i.Omega(u)", mode="el_avg")
    pass
    # return problem.evaluate("ev_grad.i.Omega(u)", mode="qp")


def stress(strain):
    # for hadamard product use np.multiply
    return np.dot(
        matcoefs.stiffness_from_youngpoisson(
            dim=2, young=E, poisson=NU, plane="strain"
        ),
        strain,
    )


def c_mod():
    return matcoefs.stiffness_from_youngpoisson(
        dim=2, young=E, poisson=NU, plane="strain"
    )


def stress_mod(strain, strain_in, damage_in):
    if np.trace(strain_in) >= 0:
        sigma = (damage_in**2 + ETA) * stress(strain)
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
    #
    # Also think about where these filters are stored and how they get passed to this function

    return sigma

    # return max(phi, phi_last)
    #
    #
    # I know the dimensions of the mesh, so actually phi doesn't have to be a field at all...
    # It can just be a matrix that's managed independantly of the state variables
    # It can be initialised pre loop and then updated in the step function
    #
    # Actuallyyyyyyy that might be hard because how do you know which element is which?
    # This will only work if the equation allows matrix multiplication
    # Remember (%s) * dw_term()""" % phi_matrix,


def load(ts, coors, **kwargs):
    print(ts.time)
    return 0.001 * ts.time


def get_energy(coors, region=None, variable=None, **kwargs):
    """
    Initialise the energy field.
    """
    print("Initialising energy field.")
    return np.zeros(
        int(
            # options["n_elements"],
            coors.shape[0],
        ),
        dtype=np.float64,
    )


def update_energy(pb, strain, stress):
    # Get the current energy field
    phi = pb.get_variables()["phi"].data[0]
    print(phi)

    for i in range(pb.domain.mesh.n_el):
        # Calculate trace ([xx] + [yy])
        trace = strain[i][0][0] + strain[i][0][1]

        # Calculate the energy in the element
        full_energy = 0.5 * np.tensordot(strain[i][0], stress[i][0])
        decomp_energy = 0.5 * np.tensordot(
            tensors.get_deviator(stress[i][0]),
            tensors.get_deviator(strain[i][0]),
        )

        # Update the energy field
        if trace >= 0.0:
            phi[i] = max(phi[i], full_energy)
        else:
            phi[i] = max(phi[i], decomp_energy)

    return phi


def pre_process(pb):
    number_elements = pb.domain.mesh.n_el
    number_nodes = pb.domain.mesh.n_nod
    options.update({"n_elements": number_elements, "n_nodes": number_nodes})
    print("The number of elements in the mesh is " + str(number_elements))
    print("The number of nodes in the mesh is " + str(number_nodes))


def step_hook(pb, ts, variables):
    print(pb.ebcs[1].dofs)
    pb.ebcs[1].dofs["u_disp.1"] = -ts.time
    strain_cur = pb.evaluate("ev_cauchy_strain.i.Omega(u_disp)", mode="el_avg")
    stress_cur = pb.evaluate("ev_cauchy_stress.i.Omega(m.C, u_disp)", mode="el_avg")
    damage_cur = pb.evaluate("ev_integrate.i.Omega(u_phase)", mode="el_avg")
    pb.get_variables()["phi"].data[0] = update_energy(pb, strain_cur, stress_cur)


def post_process(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    from sfepy.base.base import Struct
    from sfepy.mechanics.tensors import get_von_mises_stress

    ev = pb.evaluate
    stress = ev("ev_cauchy_stress.i.Omega(m.C, u_disp)", mode="el_avg")
    energy = ev("ev_integrate.i.Omega(phi)", mode="el_avg")

    vms = get_von_mises_stress(stress.squeeze())
    vms.shape = (vms.shape[0], 1, 1, 1)
    out["von_mises_stress"] = Struct(
        name="output_data", mode="cell", data=vms, dofs=None
    )
    out["elastic_energy"] = Struct(
        name="output_data", mode="cell", data=energy, dofs=None
    )

    # pb.save_state(os.path.join(pb.output_dir, "test.vtk"), out=out)

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
    "pre_process_hook": "pre_process",
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
    "phi": (
        "parameter field",
        "damage",
        {"ic": "get_energy"},
        1,
    ),
}

# may not be necessary
# material-nonlinearity.py shows that the material can be defined using a function directly
# In that example the mu value is strain dependant, however I can't tell whether it's anisotropic
materials = {
    "m": (
        {
            "C": matcoefs.stiffness_from_youngpoisson(
                dim=2, young=E, poisson=NU, plane="strain"
            ),
            "C_mod": c_mod(),
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
equations = {
    "eq_disp": """dw_lin_elastic.i.Omega(m.C_mod, v_disp, u_disp) = 0""",
    # "eq_phase": """dw_laplace.i.Omega(m.GCLS, v_phase, u_phase) + 2 * phi * dw_dot.i.Omega(v_phase, u_phase) + dw_dot.i.Omega(m.GC_LS, v_phase, u_phase) = dw_integrate.i.Omega(m.GC_LS, v_phase)""",
    # Currently running with no damage irreversibilty
    # Wierd 3rd term just to include phi for testing
    "eq_phase": """dw_laplace.i.Omega(m.GCLS, v_phase, u_phase) + 2 * dw_dot.i.Omega(v_phase, u_phase) + 2 * dw_dot.i.Omega(v_phase, phi) + dw_dot.i.Omega(m.GC_LS, v_phase, u_phase) = dw_integrate.i.Omega(m.GC_LS, v_phase)""",
}

functions = {
    "get_energy": (get_energy,),
    "load": (load,),
}

ics = {
    "phase": ("Omega", {"damage": 1.0}),
    "disp": ("Omega", {"displacement": 0.0}),
    # "energy": ("Omega", {"phi": 0.0}),
}

ebcs = {
    "fixed": ("Fixed", {"u_disp.all": 0.0}),
    "load": (
        "Load",
        {"u_disp.0": 0.0, "u_disp.1": 0.0},
    ),  # maybe set as a function of time
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
            # "dt": 0.01,  # need to make this dynamic...
            "dt": None,
            "n_step": 50,
            "quasistatic": True,
            "verbose": 1,
        },
    ),
}
