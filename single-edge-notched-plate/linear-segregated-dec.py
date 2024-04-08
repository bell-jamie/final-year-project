from __future__ import absolute_import
from datetime import datetime
from sfepy.mechanics import matcoefs, tensors
from sfepy.discrete.fem import Mesh
from sfepy.discrete.conditions import EssentialBC, Conditions
from sfepy.discrete import Problem

import numpy as np
import os


# The material properties are being calculated before the damage equation, therefore will be out of date by the time the displacement equation is solved
# This probably depends on whether my assumption of sequential solving is correct
# Given the commandline output, it seems as though it is being solved as a multifield coupled problem
# Reversing the field priority order will mean that the elastic strain energy will be out of date by the time damage is solved

# Eval equations function in Problem class might be a way to solve damage before calculating c_mod
# Orr there is the update materials function which could somehow be called after the damage equation is solved


def material(ts, coors, mode=None, **kwargs):
    """
    Calculates the modified stiffness tensor for each quadrature point.
    Returns the original stiffness tensor and the modified stiffness tensor.
    Also returns the Griffith criterion multiplied and divided by the length scale.
    """
    if mode == "qp":
        pb = kwargs["problem"]

        stn = (-1, 3, 1)
        dam = (-1, 1, 1)

        strain = pb.evaluate("ev_cauchy_strain.i.Omega(u_disp)", mode="qp").reshape(stn)
        damage = pb.evaluate("ev_integrate.i.Omega(u_phase)", mode="qp").reshape(dam)

        cns = (1, 1, 1)
        dim = (strain.shape[0], 1, 1)

        c = CMAT
        c_dev = tensors.get_deviator(c)
        c_vol = tensors.get_volumetric_tensor(c)
        c_mod = np.zeros((strain.shape[0] * strain.shape[1], c.shape[0], c.shape[1]))
        gcls = np.full(cns, GC * LS)
        gc_ls = np.full(cns, GC / LS)

        trace = strain[:, 0, 0] + strain[:, 1, 0]
        c_tens = (damage[:, 0, 0] ** 2 + ETA).reshape(dim) * c
        c_comp = (damage[:, 0, 0] ** 2 + ETA).reshape(dim) * c_dev + c_vol
        c_mod = np.where(trace.reshape(dim) >= 0, c_tens, c_comp)

        return {"C": c_mod, "GCLS": gcls, "GC_LS": gc_ls}


def energy(ts, coors, mode=None, **kwargs):
    """
    Calculates the elastic strain energy coefficient for each quadrature point.
    The elastic strain energy for each point must be monotonically increasing with time to reflect the irreversible nature of damage.
    The factor of 1/2 from the calculation of the energy has been cancelled with the factor of 2 in the weak form.
    """
    if mode != "qp":
        return

    def dev(tensor):
        trace = tensor[:, 0, 0] + tensor[:, 1, 0]
        tensor[:, 0, 0] -= trace / 2
        tensor[:, 1, 0] -= trace / 2
        return tensor

    pb = kwargs["problem"]
    phi_old = pb.phi

    s = (-1, 3, 1)

    strain = pb.evaluate("ev_cauchy_strain.i.Omega(u_disp)", mode="qp").reshape(s)
    stress = pb.evaluate("ev_cauchy_stress.i.Omega(c.C, u_disp)", mode="qp").reshape(s)

    trace = strain[:, 0, 0] + strain[:, 1, 0]
    product = np.einsum("ijk, ijk -> i", stress, strain)
    dev_product = np.einsum("ijk, ijk -> i", dev(stress), dev(strain))
    phi_new = np.where(trace >= 0, product, dev_product).reshape(phi_old.shape)
    phi = np.maximum(phi_old, phi_new)

    return {"phi": phi}


# def energy_fast(ts)


def adapt_time_step(ts, status, adt, problem, verbose=False):
    if ts.time < 5e-3:
        ts.set_time_step(1e-4)
    else:
        ts.set_time_step(1e-5)

    return True


def pre_process(pb):
    # Initialise the elastic energy - 6 quad points per element
    pb.phi = np.zeros((pb.domain.mesh.n_el * 6, 1, 1))


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
    stress = ev("ev_cauchy_stress.i.Omega(c.C, u_disp)", mode="el_avg")
    damage = ev("ev_integrate.i.Omega(u_phase)", mode="el_avg")

    vms = get_von_mises_stress(stress.squeeze())
    vms.shape = (vms.shape[0], 1, 1, 1)

    out["von_mises_stress"] = Struct(
        name="output_data", mode="cell", data=vms, dofs=None
    )
    out["damage"] = Struct(name="output_data", mode="cell", data=damage, dofs=None)

    return out  # needed?


# Constants (SI units, with mm as base length unit)
T0 = 0.0  # Initial time (always 0)
T1 = 10.0e-3  # Analogous to applied displacement (mm)
DT = 2.5e-3  # Time step
# STEPS = int(T1 / DT) // 20

E = 210e3  # Young's modulus (MPa)
NU = 0.3  # Poisson's ratio

LS = 0.0075  # Length scale (mm)
GC = 2.7  # Fracture energy (N/mm)
ETA = 1e-15
CMAT = matcoefs.stiffness_from_youngpoisson(dim=2, young=E, poisson=NU, plane="strain")

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
    "pre_process_hook": "pre_process",
    "post_process_hook": "post_process",
    "save_times": 100,
    # "block_solve": True,
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
    # "Crack": (
    #     "vertices in ((x < 0.05) & (x > -0.05) & (y < 0) & (y > -0.5))",
    #     "vertex",
    # ),
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
    "c": ({"C": CMAT},),
    "m": "material",
    "energy": "energy",
}

integrals = {
    # "i": ORDER,
    "i": DEGREE,
}

equations = {
    "eq_disp": """dw_lin_elastic.i.Omega(m.C, v_disp, u_disp) = 0""",
    "eq_phase": """dw_laplace.i.Omega(m.GCLS, v_phase, u_phase) + dw_dot.i.Omega(energy.phi, v_phase, u_phase) + dw_dot.i.Omega(m.GC_LS, v_phase, u_phase) = dw_integrate.i.Omega(m.GC_LS, v_phase)""",
}

functions = {
    "material": (material,),
    "energy": (energy,),
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
    # "crack": ("Crack", {"u_phase.all": 0.0}),
}

solvers = {
    "ls": ("ls.scipy_direct", {}),
    "newton": (
        "nls.newton",
        {
            "i_max": 20,  # should be 1 for linear
            "eps_a": 1e-6,
        },
    ),
    "ts": (
        "ts.adaptive",
        {
            "t0": T0,
            "t1": T1,
            "dt": DT,
            "dt_red_factor": 0.2,
            "dt_red_max": 0.001,
            "dt_inc_factor": 1.25,
            "dt_inc_on_iter": 4,
            "dt_inc_wait": 5,
            "verbose": 1,
            "quasistatic": True,
            "adapt_fun": adapt_time_step,
        },
    ),
}
