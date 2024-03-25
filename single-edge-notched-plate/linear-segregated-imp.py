from argparse import ArgumentParser
from datetime import datetime
from sfepy.base.base import IndexedStruct
from sfepy.mechanics import matcoefs, tensors
from sfepy.discrete import (
    FieldVariable,
    Material,
    Integral,
    Equation,
    Equations,
    Problem,
)
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton

import numpy as np
import os


def main():
    parser = ArgumentParser()  # useless?

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

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_directory = os.path.dirname(__file__)
    filename_mesh = os.path.join(
        script_directory, "meshes", "notchedPlateTriangular.vtk"
    )
    save_directory = os.path.join(
        script_directory,
        "files",
        os.path.splitext(os.path.basename(__file__))[0] + "-py",
        current_datetime,
    )

    os.makedirs(save_directory, exist_ok=True)

    mesh = Mesh.from_file(filename_mesh)
    domain = FEDomain("domain", mesh)

    omega = domain.create_region("Omega", "all")
    load = domain.create_region("Load", "vertices in (y > 0.49)", "facet")
    fixed = domain.create_region("Fixed", "vertices in (y < -0.49)", "facet")

    displacement = Field.from_args("Displacement", np.float64, 2, omega, ORDER)
    damage = Field.from_args("Damage", np.float64, 1, omega, ORDER)

    u_disp = FieldVariable("u_disp", "unknown", displacement)
    v_disp = FieldVariable("v_disp", "test", displacement, primary_var_name="u_disp")

    # history is questionable
    u_phase = FieldVariable("u_phase", "unknown", damage, history=1)
    v_phase = FieldVariable("v_phase", "test", damage, primary_var_name="u_phase")

    m = Material(
        "m",
        C=matcoefs.stiffness_from_youngpoisson(
            dim=2, young=E, poisson=NU, plane="strain"
        ),
        GCLS=GC * LS,
        GC_LS=GC / LS,
    )

    int = Integral("i", order=DEGREE)

    s1 = Term.new("dw_laplace(m.GCLS, v, u)", int, omega, m=m, v=v_phase, u=u_phase)
    s2 = Term.new("dw_dot(v, u)", int, omega, v=v_phase, u=u_phase)
    s3 = Term.new("dw_dot(m.GC_LS, v, u)", int, omega, m=m, v=v_phase, u=u_phase)
    s4 = Term.new("ev_integrate(m.GC_LS, u)", int, omega, m=m, u=u_phase)
    eq_phase = Equation("eq_phase", s1 + 2 * s2 + s3 - s4)

    u1 = Term.new("dw_lin_elastic(m.C, v, u)", int, omega, m=m, v=v_disp, u=u_disp)
    eq_disp = Equation("eq_disp", u1)

    eqs = Equations([eq_disp, eq_phase])

    fixed = EssentialBC("fixed", fixed, {"u_disp.all": 0.0})
    load = EssentialBC("load", load, {"u_disp.0": 0.0001})

    ls = ScipyDirect({})
    nls_status = IndexedStruct()
    nls = Newton({}, lin_solver=ls, status=nls_status)

    pb = Problem("phase-field", equations=eqs)
    pb.save_regions_as_groups("regions")
    pb.set_bcs(ebcs=Conditions([fixed, load]))
    pb.set_solver(nls)

    status = IndexedStruct()
    variables = pb.solve(status=status)

    print("Nonlinear solver status:\n", nls_status)
    print("Stationary solver status:\n", status)

    pb.save_state("phase-field.vtk", variables)


if __name__ == "__main__":
    main()
