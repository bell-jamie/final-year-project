from __future__ import absolute_import
from datetime import datetime
from sfepy.mechanics import matcoefs, tensors

# from sfepy.discrete import Problem
from sfepy.discrete.fem import Mesh
from sfepy.discrete.conditions import EssentialBC, Conditions
from sfepy.discrete import Problem

import numpy as np
import os

# I have an idea that I can generate filter matrices eg... [[0, 1, 0], [1, 1, 1], [0, 1, 0]
# tensile filter = generate_filter_matrix(), compressive filter = ones() - tensile filter (inverted)
# Then multiply ((damage^2 + eta) * stress tensor) and ((damage^2 + eta) * stress_dev + stress_vol) by filter matrix
# Maybe just one filter... and then tensile = filter, compression = filter.T
# The filter gets calculated in the step function and then passed to the stress_mod function
# This way the only unknown is u_disp and calculations can be faster
# Try and write this out by hand to validate the idea
# Also think about where these filters are stored and how they get passed to this function


# Right now I think that the material properties only vary over the element's quadrature points
# Additionally, the material properties are being calculated before the damage equation, therefore will be out of date by the time the displacement equation is solved
# This probably depends on whether my assumption of sequential solving is correct
# Reversing the field priority order will mean that the elastic strain energy will be out of date by the time damage is solved
#
# Eval equations function in Problem class might be a way to solve damage before calculating c_mod
# Orr there is the update materials function which could somehow be called after the damage equation is solved


# Another idea: since the filter can be a field which is 1 or 0 for each element of qp, maybe it could be inlined into one of the equations?


# ValueError: incompatible shapes! (n_qp: 6, (454, 3, 3))
# ValueError: material parameter array must have three dimensions! ('C' has 4)


def c_mod(ts, coors, mode=None, **kwargs):
    if mode == "qp":
        pb = kwargs["problem"]

        strain = pb.evaluate("ev_cauchy_strain.i.Omega(u_disp)", mode="qp")
        damage = pb.evaluate("ev_integrate.i.Omega(u_phase)", mode="qp")

        c = matcoefs.stiffness_from_youngpoisson(
            dim=2, young=E, poisson=NU, plane="strain"
        )

        c_modified = np.zeros(
            (strain.shape[0] * strain.shape[1], c.shape[0], c.shape[1])
        )

        for i in range(strain.shape[0]):
            for j in range(strain.shape[1]):
                trace = strain[i][j][0][0] + strain[i][j][1][0]

                if trace >= 0:
                    c_modified[i * strain.shape[1] + j] = (
                        damage[i][j][0] ** 2 + ETA
                    ) * c
                else:
                    c_modified[i * strain.shape[1] + j] = (
                        damage[i][j][0] ** 2 + ETA
                    ) * tensors.get_deviator(c) + tensors.get_volumetric_tensor(c)

        return {"C": c_modified}


def get_energy(coors, region=None, variable=None, **kwargs):
    """
    Initialise the energy field.
    """
    return np.zeros(
        int(
            coors.shape[0],
        ),
        dtype=np.float64,
    )


def update_energy(phi, strain, stress):
    for i in range(strain.shape[0]):
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


def update_filters(filter_qp, filter_el, strain):
    for i in range(strain.shape[0]):
        for j in range(strain.shape[1]):
            trace = strain[i][j][0] + strain[i][j][1]

            if trace >= 0:
                filter_qp[0][i * strain.shape[1] + j] = 1

            if j == 0:
                filter_el[0][i] = 1

    filter_qp[1] = 1 - filter_qp[0]
    filter_el[1] = 1 - filter_el[0]

    return (filter_qp, filter_el)


def pre_process(pb):
    number_elements = pb.domain.mesh.n_el
    number_nodes = pb.domain.mesh.n_nod

    # Initialise filters - magic number is qp per element
    qp_zeros = np.zeros(number_elements * 6, np.float64)
    el_zeros = np.zeros(number_elements)
    pb.filter_qp = [qp_zeros, qp_zeros]
    pb.filter_el = [el_zeros, el_zeros]


def step_hook(pb, ts, variables):
    # Update the load boundary condition
    pb.ebcs[1].dofs["u_disp.1"] = ts.time
    # Update the filters - currently not used
    (pb.filter_qp, pb.filter_el) = update_filters(
        pb.filter_qp,
        pb.filter_el,
        pb.evaluate("ev_cauchy_strain.i.Omega(u_disp)", mode="qp"),
    )
    # Update the elastic energy
    print("Updating elastic energy...")
    pb.get_variables()["phi"].data[0] = update_energy(
        pb.get_variables()["phi"].data[0],
        pb.evaluate("ev_cauchy_strain.i.Omega(u_disp)", mode="el_avg"),
        pb.evaluate("ev_cauchy_stress.i.Omega(m.C, u_disp)", mode="el_avg"),
    )
    print("Done!")


def post_process(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    from sfepy.base.base import Struct
    from sfepy.mechanics.tensors import get_von_mises_stress

    ev = pb.evaluate
    stress = ev("ev_cauchy_stress.i.Omega(m.C, u_disp)", mode="el_avg")
    energy = ev("ev_integrate.i.Omega(phi)", mode="el_avg")
    damage = ev("ev_integrate.i.Omega(u_phase)", mode="el_avg")

    vms = get_von_mises_stress(stress.squeeze())
    vms.shape = (vms.shape[0], 1, 1, 1)

    out["von_mises_stress"] = Struct(
        name="output_data", mode="cell", data=vms, dofs=None
    )
    out["elastic_energy"] = Struct(
        name="output_data", mode="cell", data=energy, dofs=None
    )
    out["damage"] = Struct(name="output_data", mode="cell", data=damage, dofs=None)

    return out  # needed?


# Constants
T0 = 0.0  # Initial time (always 0)
T1 = 7.0e-3  # Analogous to applied displacement (base units)
DT = 0.1  # Time step
STEPS = 30

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

# Test mesh:
# filename_mesh = data_dir + "/meshes/2d/rectangle_tri.mesh"

# For mesh conversion use:
# sfepy-convert -d 2 test.msh test.vtk
##############################

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_directory = os.path.dirname(__file__)
filename_mesh = os.path.join(script_directory, "meshes", "notchedPlateTriangularPy.vtk")
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
    "output_dir": save_directory,
    "pre_process_hook": "pre_process",
    "post_process_hook": "post_process",
}

# Should be y > 0.49 - use 8.9 for test mesh
regions = {
    "Omega": "all",
    "Load": (
        "vertices in (y > 0.49)",
        "facet",
    ),
    "Fixed": (
        "vertices in (y < -0.49)",
        "facet",
    ),
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
        0,
    ),
}

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
    "m_mod": "c_mod",
}

integrals = {
    # "i": ORDER,
    "i": DEGREE,
}

# sfepy/examples/acoustics/acoustics3d.py - has some interesting equation syntax
equations = {
    "eq_disp": """dw_lin_elastic.i.Omega(m_mod.C, v_disp, u_disp) = 0""",
    # "eq_phase": """dw_laplace.i.Omega(m.GCLS, v_phase, u_phase) + 2 * phi * dw_dot.i.Omega(v_phase, u_phase) + dw_dot.i.Omega(m.GC_LS, v_phase, u_phase) = dw_integrate.i.Omega(m.GC_LS, v_phase)""",
    # Currently running with no damage irreversibilty
    # Wierd 3rd term just to include phi for testing
    "eq_phase": """dw_laplace.i.Omega(m.GCLS, v_phase, u_phase) + 2 * dw_dot.i.Omega(v_phase, u_phase) + 2 * dw_dot.i.Omega(v_phase, phi) + dw_dot.i.Omega(m.GC_LS, v_phase, u_phase) = dw_integrate.i.Omega(m.GC_LS, v_phase)""",
}

functions = {
    "get_energy": (get_energy,),
    "c_mod": (c_mod,),
}

ics = {
    "phase": ("Omega", {"damage": 1.0}),
    "disp": ("Omega", {"displacement": 0.0}),
}

ebcs = {
    "fixed": ("Fixed", {"u_disp.all": 0.0}),
    "load": (
        "Load",
        {"u_disp.0": 0.0, "u_disp.1": 0.0},
    ),
}

solvers = {
    "ls": ("ls.scipy_direct", {}),
    "nls": (
        "nls.newton",
        {
            "i_max": 100,  # should be 1 for linear
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
            "n_step": STEPS,
            "quasistatic": True,
            "verbose": 1,
        },
    ),
}
