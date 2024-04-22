from __future__ import absolute_import
from datetime import datetime
from sfepy.discrete.problem import IndexedStruct
from sfepy.mechanics import matcoefs, tensors
from sfepy.discrete.fem import Mesh
from sfepy.discrete.conditions import EssentialBC, Conditions
from sfepy.discrete import Problem
from sfepy.base.base import Struct
from sfepy.mechanics.tensors import get_von_mises_stress

import csv
import numpy as np
import os


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
    pb.phi = phi

    return {"phi": phi}


def time_step(ts, status, adt, pb, verbose=False):
    """
    Sets the time step (load displacement) based on the current displacement.
    Also saves the displacement, damage and elastic strain energy at the current time step.
    If the solver residual is above a certain tolerance, the time step is halved and the field history is recovered.
    """
    print("Time step hook: setting displacement.")
    u = pb.ebcs[1].dofs["u_disp.1"]

    if status.err < TOL:
        # Default step size
        if u < 3e-3:
            pb.step = 1e-3
        elif u < 5e-3:
            pb.step = 1e-4
        else:
            pb.step = 1e-5

        # Update the displacement
        ts.time = pb.ebcs[1].dofs["u_disp.1"] = u + pb.step

    return True


def pre_process(pb):
    """
    Initialises the elastic energy.
    """
    print("Pre process hook: initialising elastic energy.")

    match pb.domain.mesh.descs[0]:
        case "2_3":
            quad = 6
        case "2_4":
            quad = 9
        case _:
            raise ValueError("Unsupported element type.")

    pb.phi = np.zeros((pb.domain.mesh.n_el * quad, 1, 1))
    pb.step = 0


def post_process(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    print("Post process hook: calculating stress, damage and load force.")
    disp = pb.ebcs[1].dofs["u_disp.1"]
    cells = pb.domain.mesh.n_el
    ev = pb.evaluate

    # Von mises stress
    stress = ev("ev_cauchy_stress.i.Omega(c.C, u_disp)", mode="el_avg")
    vms = get_von_mises_stress(stress.squeeze())
    vms.shape = (vms.shape[0], 1, 1, 1)
    out["vm_stress"] = Struct(name="output_data", mode="cell", data=vms, dofs=None)

    # Damage and energy
    damage = ev("ev_integrate.i.Omega(u_phase)", mode="el_avg")
    damage_sum = 1 - np.einsum("ijkl->", damage) / cells
    energy_sum = 0.5 * np.einsum("ijk->", pb.phi) / cells

    # Force
    force = ev("ev_cauchy_stress.i.Force(m.C, u_disp)", mode="eval")

    # Write to log
    with open(os.path.join(save_directory, "log.csv"), mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([disp, force[0], damage_sum, energy_sum])

    # Display stats
    print(f"\n############### STATS ###############")
    print(f"Step: {pb.ts.n_step}")
    print(f"Displacement: {1000 * disp} mm")
    print(f"Force: {force[0]} N")
    print(f"Damage: {damage_sum}")
    print(f"Energy: {energy_sum}")
    print(f"#####################################\n")

    return out


# Constants (SI units, with mm as base length unit)
T0 = 0.0  # Initial time (always 0)
T1 = 9e-3 + 1e-3  # Final displacement (always add 1e-3 because of adaptive solver)
DT = 2.5e-3  # Initial time step -- NOT USED
TOL = 1e-8  # Tolerance for the nonlinear solver
IMAX = 1  # Maximum number of solver iterations

E = 210e3  # Young's modulus (MPa)
NU = 0.3  # Poisson's ratio

LS = 0.0075  # Length scale (mm)
GC = 2.7  # Fracture energy (N/mm)
ETA = 1e-15
CMAT = matcoefs.stiffness_from_youngpoisson(dim=2, young=E, poisson=NU, plane="strain")

ORDER = 2
DEGREE = 2 * ORDER

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_directory = os.path.dirname(__file__)
filename_mesh = os.path.join(script_directory, "meshes", "notchedPlateTriNoCrack.vtk")
save_directory = os.path.join(
    script_directory,
    "files",
    os.path.splitext(os.path.basename(__file__))[0] + "-py",
    current_datetime,
)
os.makedirs(save_directory, exist_ok=True)
with open(os.path.join(save_directory, "log.csv"), mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Displacement", "Force", "Damage", "Energy"])

options = {
    "nls": "newton",
    "ls": "ls",
    "output_dir": save_directory,
    "pre_process_hook": "pre_process",
    "post_process_hook": "post_process",
    "save_times": "all",
}

regions = {
    "Omega": "all",
    "Load": (
        "vertices in (y > 0.99)",
        "vertex",
    ),
    "Fixed": (
        "vertices in (y < 0.01)",
        "vertex",
    ),
    "Force": (
        "vertices in (y > 0.99)",
        "facet",
    ),
    "Crack": (
        "vertices in (x < 0.5) & (y < 0.505) & (y > 0.495)",
        "cell",
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
    "c": ({"C": CMAT},),
    "m": "material",
    "energy": "energy",
}

integrals = {
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
    "load": ("Load", {"u_disp.1": 0.0}),
    # "load": (
    #     "Load",
    #     {"u_disp.0": 0.0, "u_disp.1": 0.0},
    # ),
    "crack": ("Crack", {"u_phase.all": 0.0}),
}

solvers = {
    "ls": ("ls.scipy_direct", {}),
    "newton": (
        "nls.newton",
        {
            "i_max": IMAX,
        },
    ),
    "ts": (
        "ts.adaptive",
        {
            "t0": T0,
            "t1": T1,
            "dt": DT,
            "quasistatic": True,
            "adapt_fun": time_step,
        },
    ),
}
