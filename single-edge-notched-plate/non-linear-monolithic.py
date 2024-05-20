from __future__ import absolute_import
from datetime import datetime
from sfepy.mechanics import matcoefs, tensors

import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import re


def material_properties(ts, coors, mode=None, **kwargs):
    """
    Function to calculate the field-dependant material properties.
    Psi = 2 * strain energy density. This removes the factor of 2 from the weak form.
    """
    if mode != "qp":
        return {}

    pb = kwargs["problem"]
    ev = lambda term: pb.evaluate(term, mode=mode, verbose=False)
    dims = pb.psi.shape

    # Evaluate and damage
    strain = ev("ev_cauchy_strain.i.Omega(u_disp)").reshape(-1, 3, 1)
    damage = ev("ev_integrate.i.Omega(u_phase)").reshape(-1, 1, 1)

    # Trace for tension or compression
    trace = strain[:, 0, 0] + strain[:, 1, 0] >= 0

    # Elastic strain energy for damage
    psi = np.maximum(
        pb.psi,
        np.where(
            trace,
            np.einsum("ijk, jk, ikl -> i", strain, CMAT, strain),
            np.einsum("ijk, jk, ikl -> i", strain, CDEV, strain),
        ).reshape(dims),
    )

    # Linear elastic stiffness tensor
    damage **= 2
    c_mod = np.where(
        trace.reshape(dims),
        (damage + ETA) * CMAT,
        (damage + ETA) * CDEV + CVOL,
    )

    # Apply bc after elastic energy calculation
    pb.ebcs["load"].dofs["u_disp.1"] = pb.disp

    return {"C": c_mod, "Psi": psi}


def time_step(ts, status, adt, pb, verbose=False):
    """
    Just a placeholder to allow the adaptive timestepper to be manually controlled.
    """
    return True


def pre_process_hook(pb):
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
            raise ValueError(f"Unsupported element type: {pb.domain.mesh.descs[0]}")

    pb.step = steps[0][1]
    pb.disp = 0.0
    pb.exit = 0
    pb.psi = np.zeros((pb.domain.mesh.n_el * n_quads, 1, 1))

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

        ents = len(region.entities[type[2]])
        print(f'Region "{region.name}" has {ents} {ents == 1 and type[0] or type[1]}')


def nls_iter_hook(pb, nls, vec, it, err, err0):
    """
    Gets called before each iteration of the nonlinear solver.
    """
    print("Iteration hook: updating materials.")
    pb.update_materials()


def post_process_hook(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    print("Post process hook: calculating stress, damage and load force.")

    disp = pb.ebcs[1].dofs["u_disp.1"]
    cells = pb.domain.mesh.n_el
    ev = pb.evaluate

    # Updating elastic energy
    pb.psi = material_properties(pb.ts, pb.domain.mesh.coors, "qp", problem=pb)["Psi"]

    # Total problem damage
    damage_new = cells - ev("ev_integrate.i.Omega(u_phase)", mode="eval")

    # Damage
    damage = ev("ev_integrate.i.Omega(u_phase)", mode="el_avg")
    damage_avg = 1 - np.einsum("ijkl->", damage) / cells
    energy_avg = 0.5 * np.einsum("ijk->", pb.psi) / cells

    # Force - [[xx], ->[yy]<-, [2xy]]
    force = ev("ev_cauchy_stress.i.Fixed(mat.C, u_disp)", mode="eval")[1]

    # Write to log
    with open(os.path.join(save_directory, "log.csv"), mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([disp, force, damage_avg, energy_avg])

    # Display stats
    print(f"\n############### STATS ###############")
    print(f"Step: {pb.ts.n_step}")
    print(f"Time: {datetime.now() - start_time}")
    print(f"Displacement: {1000 * disp} mm")
    print(f"Step size: {pb.step}")
    print(f"Force: {force} N")
    print(f"Damage: {damage_new}")
    print(f"#####################################\n")

    # Calculate next displacement step
    pb.step = steps.pop(0)[1] if steps and disp >= steps[0][0] else pb.step
    pb.ts.time = pb.disp = disp + pb.step

    # Exit criteria
    if disp >= DEX:
        print("Displacement exit criteria met.")
        pb.ts.time = T1

    if force < FEX[0] and pb.exit + FEX[1] < disp:
        print("Force exit criteria met.")
        pb.ts.time = T1
    elif force > FEX[0]:
        pb.exit = disp

    if pb.ts.n_step % 20 == 2:
        purge_savefiles()

    return out


def post_process_final_hook(problem, state):
    """
    Used for creating force-displacement plot.
    """
    with open(os.path.join(save_directory, "log.csv"), mode="r") as file:
        reader = csv.reader(file)
        data = list(reader)

    disp = [float(row[0]) for row in data[1:]]
    force = [float(row[1]) for row in data[1:]]

    fig, ax = plt.subplots()
    ax.plot(disp, force)
    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(save_directory, "force_displacement.png"))


def purge_savefiles():
    """
    Used to save space by removing all but every 10th save file.
    A better solution would be to only save the 10th save file,
    however the post_process hook must be called after each step.
    """
    mesh_name, _ = os.path.splitext(os.path.basename(filename_mesh))
    pattern = re.compile(rf"{mesh_name}\.(\d{{5}})\.vtk")

    for filename in os.listdir(save_directory):
        match = pattern.match(filename)
        if match:
            save_number = int(match.group(1))
            if save_number % 50 != 0:
                try:
                    os.remove(os.path.join(save_directory, filename))
                    print(f"Removed {filename}")
                except OSError as e:
                    print(f"Failed to remove {filename}: {e}")


# Constants (SI units, with mm as base length unit)
T0 = 0.0  # Initial time (always 0)
T1 = 1.0  # Arbitrary final time
DT = 1.0  # Initial time step -- NOT USED
TOL = 5e-11  # Tolerance for the nonlinear solver
IMAX = 1000  # Maximum number of solver iterations
FEX = (1.0, 5e-4)  # Exit force requirement (N) over which displacement (mm)
DEX = 10e-3  # Exit displacement requirement (mm) - should this be very large for sensitivity study?

E = 210e3  # Young's modulus (MPa)
NU = 0.3  # Poisson's ratio
CMAT = matcoefs.stiffness_from_youngpoisson(dim=2, young=E, poisson=NU, plane="strain")
CDEV = tensors.get_deviator(CMAT)
CVOL = tensors.get_volumetric_tensor(CMAT)

LS = 0.0075  # Length scale (mm)
GC = 2.7  # Fracture energy (N/mm)
ETA = 1e-15  # Regularization parameter

ORDER = 2
DEGREE = 2 * ORDER

MESH = "senp-damage-crack.vtk"

LOAD = {"Y": 1.0}
FIXED = {"Y": 0.0}
CRACK = {"X": 0.5, "Y1": 0.5 - LS / 5, "Y2": 0.5 + LS / 5}

steps = [
    [0, 1e-5],
    [5.3e-3, 1e-6],
]

start_time = datetime.now()
start_time_string = start_time.strftime("%Y-%m-%d_%H-%M-%S")
script_directory = os.path.dirname(__file__)
filename_mesh = os.path.join(script_directory, "meshes", MESH)
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
    "nls": "newton",
    "ls": "ls",
    "output_dir": save_directory,
    "pre_process_hook": "pre_process_hook",
    "nls_iter_hook": "nls_iter_hook",
    "post_process_hook": "post_process_hook",
    "post_process_hook_final": "post_process_final_hook",
    "save_times": "all",
}

fields = {
    "displacement": ("real", 2, "Omega", ORDER, "H1"),
    "damage": ("real", 1, "Omega", ORDER, "H1"),
}

ics = {
    "phase": ("Omega", {"damage": 1.0}),
    "disp": ("Omega", {"displacement": 0.0}),
}

variables = {
    "u_disp": ("unknown field", "displacement", 0),
    "v_disp": ("test field", "displacement", "u_disp"),
    "u_phase": ("unknown field", "damage", 1),
    "v_phase": ("test field", "damage", "u_phase"),
}

integrals = {
    "i": DEGREE,
}

equations = {
    "disp": """dw_lin_elastic.i.Omega(mat.C, v_disp, u_disp)""",
    "phase": """dw_laplace.i.Omega(const.Gcls, v_phase, u_phase)
        + dw_dot.i.Omega(mat.Psi, v_phase, u_phase)
        + dw_dot.i.Omega(const.Gc_ls, v_phase, u_phase)
        - dw_integrate.i.Omega(const.Gc_ls, v_phase)""",
}

materials = {
    "mat": "material",
    "const": ({"C": CMAT, "Gcls": GC * LS, "Gc_ls": GC / LS},),
}

functions = {
    "material": (material_properties,),
}

regions = {
    "Omega": "all",
    "Load": ("vertices in (y == %f)" % LOAD["Y"], "facet"),
    "Fixed": ("vertices in (y == %f)" % FIXED["Y"], "facet"),
    "Crack": (
        "vertices in (x <= %f) & (y >= %f) & (y <= %f)"
        % (CRACK["X"], CRACK["Y1"], CRACK["Y2"]),
        "cell",
    ),
}

regions = {
    "Omega": "all",
    "Load": ("vertices in (y == 1.0)", "facet"),
    "Fixed": ("vertices in (y == 0.0)", "facet"),
    "Crack": ("vertices in (x <= 0.5) & (y >= 0.4985) & (y <= 0.5015)", "cell"),
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
