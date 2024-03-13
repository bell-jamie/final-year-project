from __future__ import absolute_import
import numpy as np
import os

from sfepy.mechanics import matcoefs, tensors


def strain(displacement):
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


def step_hook(ts):
    pass


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

filename_mesh = os.path.join(
    os.path.dirname(__file__), "meshes", "notchedPlateTriangular.vtk"
)
save_directory = os.path.join("files-py-", os.path.splitext(__file__)[0])
os.makedirs(save_directory, exist_ok=True)

options = {
    "nls": "newton",
    "ls": "ls",
    # "step_hook": "step_hook",
    #'parametric_hook' : 'parametric_hook', # can be used to programatically change problem
    "output_dir": save_directory,
}

regions = {
    "Omega": "all",
    "Load": ("vertices in (y > 0.49)", "facet"),
    "Fixed": ("vertices in (y < -0.49)", "facet"),
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
    "phi": ("parameter field", "displacement", {"setter": "energy"}),
}

functions = {
    "strain": (strain,),
    "stress": (stress,),
    "stress_mod": (stress_mod,),
    "energy": (energy,),
}

ics = {
    "phase": ("Omega", {"damage": 1.0}),
    "disp": ("Omega", {"displacement": 0.0}),
}

ebcs = {
    "fixed": ("Fixed", {"u.all": 0.0}),
    "load": ("Load", {"u.1": 0.001}),  # maybe set as a function of time
}

# may not be necessary
materials = {
    "solid": (
        {
            "D": matcoefs.stiffness_from_youngpoisson(
                dim=2, young=E, poisson=NU, plane="strain"
            ),
        },
    ),
}

equations = {
    # "balance": """dw_integrate.i.Omega(ev_cauchy_strain.i.Omega(v_disp) * stress_mod(ev_cauchy_strain.i.Omega(u_disp), ev_cauchy_strain.i.Omega(displacement), damage)) = 0""",
    "balance": """dw_lin_elastic.i.Omega(solid.D, v_disp, u_disp) = 0""",
    # "damage": """dw_laplace.i.Omega(GC * LS, u_phase, v_phase) + dw_integrate.i.Omega(2 * s * v_phase * phi) + dw_integrate.i.Omega((GC * u_phase * v_phase)/LS) - dw_integrate.i.Omega((GC * v_phase)/LS) = 0""",
    "damage": """dw_laplace.i.Omega(v_phase, u_phase) = 0""",
}

integrals = {
    "i": ORDER,
    #'i': 2 * ORDER,
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
