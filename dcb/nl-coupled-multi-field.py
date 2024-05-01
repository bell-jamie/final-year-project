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
    """
    if mode != "qp":
        return

    pb = kwargs["problem"]
    ev = pb.evaluate
    dims = pb.phi.shape

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
    pb.phi = np.maximum(
        pb.phi,
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

    return {"C": c_mod, "phi": pb.phi}


def material_constants(ts, coors, mode=None, **kwargs):
    """
    Outputs the material constants for the problem.
    The fracture toughness is initialised before the problem is solved.
    """
    if mode != "qp":
        return

    # return {
    #     "C": np.full((qps, 3, 3), CMAT),
    #     "GCLS": np.full((qps, 1, 1), GC_I * LS),
    #     "GC_LS": np.full((qps, 1, 1), GC_I / LS),
    # }


def time_step(ts, status, adt, pb, verbose=False):
    """
    Just a placeholder to allow the adaptive timestepper to be manually controlled.
    """
    return True


def pre_process(pb):
    """
    Initialises the elastic energy.
    """
    print("Pre process hook: initialising elastic energy.")

    match pb.domain.mesh.descs[0]:
        case "2_3":
            n_quads = 6
        case "2_4":
            n_quads = 9
        case _:
            raise ValueError("Unsupported element type.")

    pb.step = 0
    pb.disp = 0
    pb.exit = 0
    pb.cutbacks = 0
    pb.phi = np.zeros((pb.domain.mesh.n_el * n_quads, 1, 1))
    pb.log = IndexedStruct(disp=[], force=[], damage=[], energy=[])
    pb.history = IndexedStruct


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

    # Cutback
    # if pb.status.nls_status.err > TOL2:
    #     # Halve time step
    #     pb.step /= 2
    #     pb.ts.time = pb.ebcs[1].dofs["u_disp.1"] = disp - pb.step

    #     # Recover history
    #     pb.get_variables()["u_disp"].data[0] = pb.history.u_disp
    #     pb.get_variables()["u_phase"].data[0] = pb.history.u_phase
    #     pb.phi = pb.history.phi

    #     pb.cutbacks += 1
    #     return out  # Unfortunately this will still save a .vtk file
    # else:
    #     # Save history
    #     pb.history.u_disp = pb.get_variables()["u_disp"].data[0]
    #     pb.history.u_phase = pb.get_variables()["u_phase"].data[0]
    #     pb.history.phi = pb.phi

    # Von mises stress
    stress = ev("ev_cauchy_stress.i.Omega(consts.C, u_disp)", mode="el_avg")
    vms = get_von_mises_stress(stress.squeeze()).reshape(stress.shape[0], 1, 1, 1)
    out["vm_stress"] = Struct(name="output_data", mode="cell", data=vms, dofs=None)

    # Damage
    damage = ev("ev_integrate.i.Omega(u_phase)", mode="el_avg")
    damage_sum = 1 - np.einsum("ijkl->", damage) / cells
    energy_sum = 0.5 * np.einsum("ijk->", pb.phi) / cells

    # Force - [[xx], ->[yy]<-, [2xy]
    force = ev("ev_cauchy_stress.i.Load(mat.C, u_disp)", mode="eval")[1]

    # Set displacement step (don't apply it to EBC yet)
    pb.step = steps.pop(0)[1] if steps and disp >= steps[0][0] else pb.step
    pb.ts.time = pb.disp = disp + pb.step

    # Apply displacement here for now...
    pb.ebcs[1].dofs["u_disp.1"] = pb.disp

    # Write to log
    with open(os.path.join(save_directory, "log.csv"), mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([disp, force, damage_sum, energy_sum])

    pb.log.disp.append(disp)
    pb.log.force.append(force)
    pb.log.damage.append(damage_sum)
    pb.log.energy.append(energy_sum)

    # Display stats
    print(f"\n############### STATS ###############")
    print(f"Step: {pb.ts.n_step}")
    print(f"Time: {datetime.now() - start_time}")
    print(f"Displacement: {1000 * disp} mm")
    print(f"Force: {force} N")
    print(f"Damage: {damage_sum}")
    print(f"Energy: {energy_sum}")
    print(f"#####################################\n")

    # Exit criteria
    if disp >= DEX:
        print("Displacement exit criteria met.")
        pb.ts.time = T1

    return out


def post_process_final(problem, state):
    """
    Used for creating force-displacement plot.
    """
    print("Number of cutbacks:" + str(problem.cutbacks))

    fig, ax = plt.subplots()
    ax.plot(problem.log.disp, problem.log.force)
    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(save_directory, "force_displacement.png"))


# Constants (SI units)
T0 = 0.0  # Initial time (always 0)
T1 = 1.0  # Arbitrary final time
DT = 2.5e-3  # Initial time step -- NOT USED
TOL = 1e-6  # Tolerance for the nonlinear solver
IMAX = 10  # Maximum number of solver iterations
DEX = 2e-3  # Exit displacement requirement (m)

E = 126e9  # Young's modulus (Pa)
NU = 0.3  # Poisson's ratio

LS = 0.03e-3  # Length scale (m)
GC_I = 281  # Interface racture energy (N/m)
GC_B = GC_I * 10  # Beam fracture energy (N/m)
ETA = 1e-8
CMAT = matcoefs.stiffness_from_youngpoisson(dim=2, young=E, poisson=NU, plane="strain")
CDEV = tensors.get_deviator(CMAT)
CVOL = tensors.get_volumetric_tensor(CMAT)

ORDER = 2
DEGREE = 2 * ORDER

steps = [
    [0, 1e-5],
    [1e-3, 1e-6],
]

start_time = datetime.now()
start_time_string = start_time.strftime("%Y-%m-%d_%H-%M-%S")
script_directory = os.path.dirname(__file__)
filename_mesh = os.path.join(script_directory, "meshes", "dcb.vtk")
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
    "pre_process_hook": "pre_process",
    "nls_iter_hook": "nls_iter_hook",
    "post_process_hook": "post_process",
    "post_process_hook_final": "post_process_final",
    "save_times": "all",
    "verbose": False,  # Does this even work?
}

regions = {
    "Omega": "all",
    "Load": (
        "vertices in (x > 0.099) & (y > 0.00276)",
        "facet",
    ),
    "Fixed": (
        "vertices in (x > 0.099) & (y < 0.00266)",
        "facet",
    ),
    "Crack": (
        "vertices in (x > 0.07) & (y < 0.00273) & (y > 0.00269)",
        "cell",
    ),
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

materials = {
    "mat": "material",
    "consts": ({"C": CMAT, "GCLS": GC_I * LS, "GC_LS": GC_I / LS},),
}

integrals = {
    "i": DEGREE,
}

equations = {
    "eq_disp": """dw_lin_elastic.i.Omega(mat.C, v_disp, u_disp) = 0""",
    "eq_phase": """dw_laplace.i.Omega(consts.GCLS, v_phase, u_phase) +
        dw_dot.i.Omega(mat.phi, v_phase, u_phase) +
        dw_dot.i.Omega(consts.GC_LS, v_phase, u_phase) =
        dw_integrate.i.Omega(consts.GC_LS, v_phase)""",
}

functions = {
    "material": (material_properties,),
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
