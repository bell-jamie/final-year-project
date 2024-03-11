from __future__ import absolute_import
import numpy as np
import os

from sfepy.mechanics import matcoefs, tensors


def stress(strain):
    return strain * matcoefs.stiffness_from_youngpoisson(
        dim=2, young=E, poisson=NU, plane="strain"
    )


def stress_mod(strain, strain_in, damage_in):
    if np.trace(strain_in) >= 0:
        stress = (damage_in**2 + ETA) * stress(strain)
    else:
        stress = tensors.get_deviator(
            damage_in**2 + ETA
        ) + tensors.get_volumetric_tensor(stress(strain))
    return stress


def positive_energy(strain):
    if np.trace(strain) >= 0:
        energy = 0.5 * (strain * stress(strain))
    else:
        energy = (
            0.5
            * tensors.get_deviator(stress(strain))
            * tensors.get_volumetric_tensor(strain)
        )
    return energy


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
    "step_hook": "step_hook",
    #'parametric_hook' : 'parametric_hook', # can be used to programatically change problem
    "output_dir": os.path.join(
        "files-py-", os.path.splitext(__file__)[0]
    ),  # change this
}

regions = {
    "Omega": "all",
    "Load": ("vertices in (y > 0.49)", "facet"),
    "Fixed": ("vertices in (y < -0.49)", "facet"),
}

materials = {
    "solid": (
        {
            "D": matcoefs.stiffness_from_youngpoisson(
                dim=2, young=E, poisson=NU, plane="strain"
            )
        }
    ),
}

fields = {
    "displacement": ("real", "vector", "Omega", ORDER, "H1"),
    "damage": ("real", "scalar", "Omega", ORDER, "H1"),
}

integrals = {
    "i": ORDER,
    #'i': 2 * ORDER,
}

variables = {
    "u_disp": ("unknown field", "displacement", 0),
    "v_disp": ("test field", "displacement", "u_disp"),
    "u_phase": (
        "unknown field",
        "damage",
        1,
    ),  # remember previous damage - history function?
    "v_phase": ("test field", "damage", "u_phase"),
    "phi": ("parameter field", "energy", {"setter": "positive_energy"}),
}

functions = {
    #'stress' : (lambda strain : strain * matcoefs.stiffness_from_youngpoisson(dim = 2, young = E, poisson = NU, plane = 'strain')),
    "stress": (stress),
    "stress_mod": (stress_mod),
    "positive_energy": (positive_energy),
}

# These are both wrong
equations = {
    "balance": """dw_lin_elastic(solid.D, v_disp, u_disp) = 0""",
    #'damage' : """dw_dot.i.Omega(GC, LS, nabla(u_phase), nabla(v_phase)) + dw_dot.""",
    "damage": """dw_laplace.i.Omega(np.dot(GC, LS), u_phase, v_phase) + """,
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

ebcs = {
    "fixed": ("Fixed", {"u.all": 0.0}),
    "load": ("Load", {"u.1": 0.001}),
}

# if __name__ == "__main__":
#   main()
