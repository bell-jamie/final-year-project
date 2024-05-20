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
    pb.ebcs["load_a"].dofs["u_disp.1"] = pb.disp / 2
    pb.ebcs["load_b"].dofs["u_disp.1"] = -pb.disp / 2

    # On the first step, use the quad point coordinates to map the interface toughness
    if pb.gc_set is False:
        map_gc(coors, pb.gc)
        pb.gc_set = True

    return {"C": c_mod, "Psi": psi, "Gcls": pb.gc * LS, "Gc_ls": pb.gc / LS}


def map_gc(coors, gc):
    """
    Maps the interface fracture toughness to the interface elements.
    """
    n_int = 0
    for i, coor in enumerate(coors):
        if coor[1] <= LOAD["YA"] and coor[1] >= LOAD["YB"]:
            gc[i] = GC_I
            n_int += 1

    print(f"Number of quads with interface fracture toughness: {n_int}")
    print(f"Number of quads with bulk fracture toughness: {len(gc) - n_int}")


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
    pb.psi = np.zeros((pb.domain.mesh.n_el * n_quads, 1, 1))
    pb.gc = np.full((pb.domain.mesh.n_el * n_quads, 1, 1), GC_B)
    pb.gc_set = False

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

    # Force - [[xx], ->[yy]<-, [2xy]] - N = Pa * m^2
    force = abs(ev("ev_cauchy_stress.i.LoadA(mat.C, u_disp)", mode="eval")[1]) * 1000

    # Write to log
    with open(os.path.join(save_directory, "log.csv"), mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([pb.disp, force, damage_avg, energy_avg])

    # Display stats
    print(f"\n############### STATS ###############")
    print(f"Step: {pb.ts.n_step}")
    print(f"Time: {datetime.now() - start_time}")
    print(f"Displacement: {1000 * pb.disp} mm")
    print(f"Step size: {pb.step}")
    print(f"Force: {force} N")
    print(f"Damage: {damage_new}")
    print(f"#####################################\n")

    # Calculate next displacement step
    pb.step = steps.pop(0)[1] if steps and pb.disp >= steps[0][0] else pb.step
    pb.disp += pb.step
    pb.ts.time = pb.disp

    # Exit criteria
    if pb.disp >= DEX:
        print("Displacement exit criteria met.")
        pb.ts.time = T1

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

    disp = [float(row[0]) * 1000 for row in data[1:]]
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


# Constants (SI units)
T0 = 0.0  # Initial time (always 0)
T1 = 1.0  # Arbitrary final time
DT = 1.0  # Initial time step -- NOT USED
TOL = 1.5e-5  # Tolerance for the nonlinear solver
IMAX = 20  # Maximum number of solver iterations
DEX = 0.35e-3  # Exit displacement requirement (m)

E = 126e9  # Young's modulus (Pa)
NU = 0.3  # Poisson's ratio
CMAT = matcoefs.stiffness_from_youngpoisson(dim=2, young=E, poisson=NU, plane="strain")
CDEV = tensors.get_deviator(CMAT)
CVOL = tensors.get_volumetric_tensor(CMAT)

LS = 30e-6  # Length scale (m)
GC_I = 187  # 281  # Interface fracture energy (N/m)
GC_B = 281 * 10  # Beam fracture energy (N/m)
ETA = 1e-15  # Regularization parameter

ORDER = 2
DEGREE = 2 * ORDER

MESH = "sym-dcb.vtk"

LOAD = {"X": 50e-3, "YA": 1.6e-3, "YB": 1.5e-3}
FIXED = {"X": 50e-3}
CRACK = {"X": 40e-3, "Y1": 1.55e-3 - LS / 2, "Y2": 1.55e-3 + LS / 2}

steps = [
    [0.15e-3, 1e-6],
    [0, 5e-8],
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
    "phase": """dw_laplace.i.Omega(mat.Gcls, v_phase, u_phase)
        + dw_dot.i.Omega(mat.Psi, v_phase, u_phase)
        + dw_dot.i.Omega(mat.Gc_ls, v_phase, u_phase)
        - dw_integrate.i.Omega(mat.Gc_ls, v_phase)""",
}

materials = {
    "mat": "material",
    "const": ({"C": CMAT},),
}

functions = {
    "material": (material_properties,),
}

regions = {
    "Omega": "all",
    "LoadA": ("vertices in (x == %f) & (y >= %f)" % (LOAD["X"], LOAD["YA"]), "facet"),
    "LoadB": ("vertices in (x == %f) & (y <= %f)" % (LOAD["X"], LOAD["YB"]), "facet"),
    "Fixed": ("vertices in (x == 0.0)", "facet"),
    "Crack": (
        "vertices in (x >= %f) & (y >= %f) & (y <= %f)"
        % (CRACK["X"], CRACK["Y1"], CRACK["Y2"]),
        "cell",
    ),
}

ebcs = {
    "fixed": ("Fixed", {"u_disp.0": 0.0}),
    "load_a": ("LoadA", {"u_disp.1": 0.0}),
    "load_b": ("LoadB", {"u_disp.1": 0.0}),
    "crack": ("Crack", {"u_phase.all": 0.0}),
}

lcbs = {
    "load_a_rigid": ("LoadA", {"u_disp.all", None}, None, "rigid"),
    "load_b_rigid": ("LoadB", {"u_disp.all", None}, None, "rigid"),
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
