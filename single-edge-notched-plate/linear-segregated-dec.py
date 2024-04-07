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


def c_mat(ts, coors, mode=None, **kwargs):
    """
    Calculates the modified stiffness tensor for each quadrature point.
    """
    if mode == "qp":
        pb = kwargs["problem"]

        strain = pb.evaluate("ev_cauchy_strain.i.Omega(u_disp)", mode="qp")
        damage = pb.evaluate("ev_integrate.i.Omega(u_phase)", mode="qp")

        c = matcoefs.stiffness_from_youngpoisson(
            dim=2, young=E, poisson=NU, plane="strain"
        )

        c_mod = np.zeros((strain.shape[0] * strain.shape[1], c.shape[0], c.shape[1]))

        for i in range(strain.shape[0]):
            for j in range(strain.shape[1]):
                trace = strain[i][j][0][0] + strain[i][j][1][0]

                if trace >= 0:
                    c_mod[i * strain.shape[1] + j] = (damage[i][j][0] ** 2 + ETA) * c
                else:
                    c_mod[i * strain.shape[1] + j] = (
                        damage[i][j][0] ** 2 + ETA
                    ) * tensors.get_deviator(c) + tensors.get_volumetric_tensor(c)

        return {"C": c_mod}


def phi_mat(ts, coors, mode=None, **kwargs):
    """
    Calculates the elastic strain energy coefficient for each quadrature point.
    The elastic strain energy for each point must be monotonically increasing with time to reflect the irreversible nature of damage.
    The factor of 1/2 from the calculation of the energy has been cancelled with the factor of 2 in the weak form.
    """
    if mode == "qp":
        pb = kwargs["problem"]

        strain = pb.evaluate("ev_cauchy_strain.i.Omega(u_disp)", mode="qp")
        stress = pb.evaluate("ev_cauchy_stress.i.Omega(m.C, u_disp)", mode="qp")

        phi = np.zeros((strain.shape[0] * strain.shape[1], 1, 1))

        for i in range(strain.shape[0]):
            for j in range(strain.shape[1]):
                trace = strain[i][j][0][0] + strain[i][j][1][0]

                if trace >= 0:
                    phi[i * strain.shape[1] + j] = max(
                        phi[i * strain.shape[1] + j],
                        np.tensordot(strain[i][j], stress[i][j]),
                    )
                else:
                    phi[i * strain.shape[1] + j] = max(
                        phi[i * strain.shape[1] + j],
                        np.tensordot(
                            tensors.get_deviator(stress[i][j]),
                            tensors.get_deviator(strain[i][j]),
                        ),
                    )

        return {"phi": phi}


def step_hook(pb, ts, variables):
    # Update the load boundary condition
    pb.ebcs[1].dofs["u_disp.1"] = ts.time


def post_process(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    from sfepy.base.base import Struct
    from sfepy.mechanics.tensors import get_von_mises_stress

    ev = pb.evaluate
    stress = ev("ev_cauchy_stress.i.Omega(m.C, u_disp)", mode="el_avg")
    # energy = ev("ev_integrate.i.Omega(phi)", mode="el_avg")
    damage = ev("ev_integrate.i.Omega(u_phase)", mode="el_avg")

    vms = get_von_mises_stress(stress.squeeze())
    vms.shape = (vms.shape[0], 1, 1, 1)

    out["von_mises_stress"] = Struct(
        name="output_data", mode="cell", data=vms, dofs=None
    )
    # out["elastic_energy"] = Struct(
    #     name="output_data", mode="cell", data=energy, dofs=None
    # )
    out["damage"] = Struct(name="output_data", mode="cell", data=damage, dofs=None)

    return out  # needed?


# Constants (SI units, with mm as base length unit)
T0 = 0.0  # Initial time (always 0)
T1 = 10.0e-3  # Analogous to applied displacement (mm)
DT = 1e-5  # Time step
STEPS = 50  # T1 / DT

E = 210e3  # Young's modulus (MPa)
NU = 0.3  # Poisson's ratio

LS = 0.0075  # Length scale (mm)
GC = 2.7  # Fracture energy (N/mm)
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
filename_mesh = os.path.join(script_directory, "meshes", "notchedPlateTriangular.vtk")
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
    # "pre_process_hook": "pre_process",
    "post_process_hook": "post_process",
}

# For test mesh, use > 8.9, < -8.9
# For notchedPlateTriangular, use > 0.49, < -0.49
# For notchedPlateRahaman, use > 0.99, < 0.01
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
    "m_mod": "c_mat",
    "energy": "phi_mat",
}

integrals = {
    # "i": ORDER,
    "i": DEGREE,
}

equations = {
    "eq_disp": """dw_lin_elastic.i.Omega(m_mod.C, v_disp, u_disp) = 0""",
    "eq_phase": """dw_laplace.i.Omega(m.GCLS, v_phase, u_phase) + dw_dot.i.Omega(energy.phi, v_phase, u_phase) + dw_dot.i.Omega(m.GC_LS, v_phase, u_phase) = dw_integrate.i.Omega(m.GC_LS, v_phase)""",
}

functions = {
    "c_mat": (c_mat,),
    "phi_mat": (phi_mat,),
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
            "i_max": 10,  # should be 1 for linear
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
