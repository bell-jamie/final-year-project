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
import matplotlib.pyplot as plt
import os


def material_properties(ts, coors, mode=None, **kwargs):
    """
    Function to calculate the field-dependant material properties.
    Psi = 2 * strain energy density. This removes the factor of 2 from the weak form.
    """
    if mode != "qp":
        return

    pb = kwargs["problem"]
    ev = pb.evaluate
    dims = pb.psi.shape

    # Evaluate and damage
    strain = ev("ev_cauchy_strain.i.Omega(u_disp)", mode=mode).reshape(-1, 3, 1)
    damage = ev("ev_integrate.i.Omega(u_phase)", mode=mode).reshape(-1, 1, 1)

    # Trace and trace mask for tensile or compressive
    trace = strain[:, 0, 0] + strain[:, 1, 0]
    trace_mask = trace >= 0

    # Deviatoric strain calculation
    strain_dev = strain.copy()
    strain_dev[:, :2, 0] -= trace[:, None] / 2

    # Elastic strain energy for damage
    pb.psi = np.maximum(
        pb.psi,
        np.where(
            trace_mask,
            np.einsum("jk, ijk, ikl -> i", CMAT, strain, strain),
            np.einsum("jk, ijk, ikl -> i", CMAT, strain_dev, strain_dev),
        ).reshape(dims),
    )

    # Linear elastic stiffness tensor
    damage **= 2
    c_mod = np.where(
        trace_mask.reshape(dims),
        (damage + ETA) * CMAT,
        (damage + ETA) * CDEV + CVOL,
    )

    return {"C": c_mod, "psi": pb.psi}


def time_step(ts, status, adt, pb, verbose=False):
    """
    Just a placeholder to allow the adaptive timestepper to be manually controlled.
    """
    return True


# def time_step(ts, status, adt, pb, verbose=False):
#     """
#     Sets the time step (load displacement) based on the current displacement.
#     Also saves the displacement, damage and elastic strain energy at the current time step.
#     If the solver residual is above a certain tolerance, the time step is halved and the field history is recovered.
#     """
#     return True

# print("Time step hook: setting displacement.")
# u = pb.ebcs[1].dofs["u_disp.1"]

# # if status.err > TOL:
# #     # Backtrack
# #     u -= pb.step  # remove the step
# #     pb.step /= 2  # halve the step

# #     # Recover from field history
# #     pb.get_variables()["u_disp"].data[0] = pb.history.u_disp
# #     pb.get_variables()["u_phase"].data[0] = pb.history.u_phase
# #     pb.psi = pb.history.psi
# # else:
# # Default step size

# # Get next time step
# if steps and u >= steps[0][0]:
#     pb.step = steps.pop(0)[1]

# # # Save field history
# # pb.history.u_disp = pb.get_variables()["u_disp"].data[0]
# # pb.history.u_phase = pb.get_variables()["u_phase"].data[0]
# # pb.history.psi = pb.psi

# # Update the displacement
# ts.time = pb.ebcs[1].dofs["u_disp.1"] = u + pb.step

# return True


def pre_process(pb):
    """
    Initialises all of the auxiliary variables for the problem.
    Uses number of quad points to initialise quad point evaluated variables.
    Checks and outputs number of entities in each region.
    """
    print("Pre process hook: initialising aux variables and checking regions.")

    match pb.domain.mesh.descs[0]:
        case "2_3":
            n_quads = 6
        case "2_4":
            n_quads = 9
        case _:
            raise ValueError("Unsupported element type.")

    for region in pb.domain.regions:
        match region.true_kind:
            case "vertex":
                type = ("vertex", "vertices", 0)
            case "edge":
                type = ("edge", "edges", 1)
            case "cell":
                type = ("cell", "cells", 2)
            case _:
                type = ("entity", "entities", 0)

        entities = len(region.entities[type[2]])

        print(
            f'Region "{region.name}" has {entities} {entities == 1 and type[0] or type[1]}'
        )

    pb.step = 0.0
    pb.disp = 0.0
    pb.exit = 0
    pb.cutbacks = 0
    pb.psi = np.zeros((pb.domain.mesh.n_el * n_quads, 1, 1))
    pb.log = IndexedStruct(disp=[], force=[], damage=[], damage_avg=[], energy_avg=[])
    pb.history = IndexedStruct


# def step_hook(pb, ts, variables):
#     """
#     Gets called after each step, right before the post process hook.
#     """
#     pass
#     print("Step hook: updating load boundary condition.")
#     pb.ebcs[1].dofs["u_disp.1"] = ts.time


def nls_iter_hook(pb, nls, vec, it, err, err0):
    """
    Gets called before each iteration of the nonlinear solver.
    """
    print("Iteration hook: updating materials.")
    pb.update_materials()


def post_process(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    print("Post process hook: calculating stress, damage and load force.")

    disp = pb.ebcs[1].dofs["u_disp.1"]
    cells = pb.domain.mesh.n_el
    ev = pb.evaluate

    # Total problem damage
    damage_new = cells - ev("ev_integrate.i.Omega(u_phase)", mode="eval")

    # Cutback
    # if pb.status.nls_status.err > TOL2:
    #     # Halve time step
    #     pb.step /= 2
    #     pb.ts.time = pb.ebcs[1].dofs["u_disp.1"] = disp - pb.step

    #     # Recover history
    #     pb.get_variables()["u_disp"].data[0] = pb.history.u_disp
    #     pb.get_variables()["u_phase"].data[0] = pb.history.u_phase
    #     pb.psi = pb.history.psi

    #     pb.cutbacks += 1
    #     return out  # Unfortunately this will still save a .vtk file
    # else:
    #     # Save history
    #     pb.history.u_disp = pb.get_variables()["u_disp"].data[0]
    #     pb.history.u_phase = pb.get_variables()["u_phase"].data[0]
    #     pb.history.psi = pb.psi

    # Apply displacement step
    pb.step = steps.pop(0)[1] if steps and disp >= steps[0][0] else pb.step
    pb.ts.time = pb.disp = disp + pb.step
    pb.ebcs["load"].dofs["u_disp.1"] = pb.disp

    # # Von mises stress
    # stress = ev("ev_cauchy_stress.i.Omega(c.C, u_disp)", mode="el_avg")
    # vms = get_von_mises_stress(stress.squeeze()).reshape(stress.shape[0], 1, 1, 1)
    # out["vm_stress"] = Struct(name="output_data", mode="cell", data=vms, dofs=None)

    # Damage
    damage = ev("ev_integrate.i.Omega(u_phase)", mode="el_avg")
    damage_avg = 1 - np.einsum("ijkl->", damage) / cells
    energy_avg = 0.5 * np.einsum("ijk->", pb.psi) / cells

    # Force - [[xx], ->[yy]<-, [2xy]
    force = ev("ev_cauchy_stress.i.Fixed(mat.C, u_disp)", mode="eval")[1]

    # Write to log
    with open(os.path.join(save_directory, "log.csv"), mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([disp, force, damage_avg, energy_avg])

    pb.log.disp.append(disp)
    pb.log.force.append(force)
    pb.log.damage.append(damage_new)
    pb.log.damage_avg.append(damage_avg)
    pb.log.energy_avg.append(energy_avg)

    # Display stats
    print(f"\n############### STATS ###############")
    print(f"Step: {pb.ts.n_step}")
    print(f"Time: {datetime.now() - start_time}")
    print(f"Displacement: {1000 * disp} mm")
    print(f"Step size: {pb.step}")
    print(f"Cutbacks: {pb.cutbacks}")
    print(f"Force: {force} N")
    print(f"Damage: {damage_new}")
    print(f"Damage avg: {damage_avg}")
    print(f"Energy avg: {energy_avg}")
    print(f"#####################################\n")

    # Exit criteria
    if disp >= DEX:
        print("Displacement exit criteria met.")
        pb.ts.time = T1

    if force < FEX[0] and pb.exit + FEX[1] < disp:
        print("Force exit criteria met.")
        pb.ts.time = T1
    elif force > FEX[0]:
        pb.exit = disp

    return out


def post_process_final(problem, state):
    """
    Used for creating force-displacement plot.
    """
    fig, ax = plt.subplots()
    ax.plot(problem.log.disp, problem.log.force)
    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(save_directory, "force_displacement.png"))


# Constants (SI units, with mm as base length unit)
T0 = 0.0  # Initial time (always 0)
T1 = 1.0  # Arbitrary final time
DT = 2.5e-3  # Initial time step -- NOT USED
TOL = 1e-10  # Tolerance for the nonlinear solver
TOL2 = 1e-11  # Tolerance for the cutback
IMAX = 10  # Maximum number of solver iterations
FEX = (1.0, 5e-4)  # Exit force requirement (N) over which displacement (mm)
DEX = 10e-3  # Exit displacement requirement (mm) - should this be very large for sensitivity study?

E = 210e3  # Young's modulus (MPa)
NU = 0.3  # Poisson's ratio
CMAT = matcoefs.stiffness_from_youngpoisson(dim=2, young=E, poisson=NU, plane="strain")
CDEV = tensors.get_deviator(CMAT)
CVOL = tensors.get_volumetric_tensor(CMAT)

LS = 0.0075  # Length scale (mm)
GC = 2.7  # Fracture energy (N/mm)
ETA = 1e-8

ORDER = 2
DEGREE = 2 * ORDER

steps = [
    [0, 1e-4],
    [5e-3, 1e-5],
]

start_time = datetime.now()
start_time_string = start_time.strftime("%Y-%m-%d_%H-%M-%S")
script_directory = os.path.dirname(__file__)
filename_mesh = os.path.join(script_directory, "meshes", "notchedPlateTriNoCrack.vtk")
save_directory = os.path.join(
    script_directory,
    "files",
    os.path.splitext(os.path.basename(__file__))[0] + "-py",
    start_time_string,
)
os.makedirs(save_directory, exist_ok=True)

# Create log file
with open(os.path.join(save_directory, "log.csv"), mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Displacement", "Force", "Damage", "Energy"])

# Read script contents
with open(os.path.realpath(__file__), mode="r") as file:
    script_content = file.read()

# Save script contents
with open(os.path.join(save_directory, "nl-coupled-multi-field.py"), mode="w") as file:
    file.write(script_content)

options = {
    "nls": "scipy",
    "ls": "ls",
    "output_dir": save_directory,
    "pre_process_hook": "pre_process",
    "nls_iter_hook": "nls_iter_hook",
    "post_process_hook": "post_process",
    "post_process_hook_final": "post_process_final",
    "save_times": "all",
}

fields = {
    "displacement": ("real", 2, "Omega", ORDER, "H1"),
    "damage": ("real", 1, "Omega", ORDER, "H1"),
}

# N.B. Order: phase-field -> displacement
variables = {
    "u_disp": ("unknown field", "displacement", 1),
    "v_disp": ("test field", "displacement", "u_disp"),
    "u_phase": (
        "unknown field",
        "damage",
        0,
    ),
    "v_phase": ("test field", "damage", "u_phase"),
}

integrals = {
    "i": DEGREE,
}

equations = {
    "eq_disp": """dw_lin_elastic.i.Omega(mat.C, v_disp, u_disp) = 0""",
    "eq_phase": """dw_laplace.i.Omega(const.GCLS, v_phase, u_phase) +
        dw_dot.i.Omega(mat.psi, v_phase, u_phase) +
        dw_dot.i.Omega(const.GC_LS, v_phase, u_phase) =
        dw_integrate.i.Omega(const.GC_LS, v_phase)""",
}

materials = {
    "mat": "material",
    "const": ({"C": CMAT, "GCLS": GC * LS, "GC_LS": GC / LS},),
}

functions = {
    "material": (material_properties,),
}

regions = {
    "Omega": "all",
    "Load": (
        "vertices in (y > 0.99)",
        "facet",
    ),
    "Fixed": (
        "vertices in (y < 0.01)",
        "facet",
    ),
    "Crack": (
        "vertices in (x < 0.5) & (y < %f) & (y > %f)" % (0.5 + LS / 5, 0.5 - LS / 5),
        "cell",
    ),
}

ics = {
    "phase": ("Omega", {"damage": 1.0}),
    "disp": ("Omega", {"displacement": 0.0}),
}

ebcs = {
    "fixed": ("Fixed", {"u_disp.all": 0.0}),
    "load": ("Load", {"u_disp.1": 0.0}),
    "crack": ("Crack", {"u_phase.all": 0.0}),
}

solvers = {
    "ls": ("ls.scipy_direct", {}),
    "newton": (
        "nls.newton",
        {
            "i_max": IMAX,
            "eps_a": TOL,
        },
    ),
    "scipy": (
        "nls.scipy_fmin_like",
        {
            "method": "fmin_bfgs",
            "i_max": IMAX,
        },
    ),
    "ts": (
        "ts.adaptive",
        {
            "t0": T0,
            "t1": T1,
            "dt": DT,
            "verbose": 0,
            "quasistatic": True,
            "adapt_fun": time_step,
        },
    ),
}
