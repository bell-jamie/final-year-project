# Phase-Field Method for Delamination Using Gridap and SfePy

## About

This project implements the mathematical formulation of the phase-field fracture model derived by Miehe et al.[^1] in [Gridap](https://github.com/gridap/Gridap.jl) and [SfePy](https://github.com/sfepy/sfepy).

## Contents

### Julia & Gridap

This implementation is used to solve the single edge notched plate tension benchmark.

- **Linear Segregated**
	- Reformatted implementation from Rahaman [^2]
	- Not accurate for non-linear unstable problems
- **Non-Linear Alternation Minimisation**
	- Uses Newton's method to and line searches to correctly capture unstable behaviour
	- Internal minimisation cycles solve each field independently
- **Non-Linear Monolithic**
	- Requires small step sizes to converge due to the non-convex nature of the coupled functional
- Functions Library
	- pfm-lib.jl is a centralised collection of the solvers and loops to avoid code repetition
	- This file is imported by the problem scripts for access to the custom phase and displacement structs and all of the helper functions

### Python & Sfepy

This implementation is used to solve both the single edge notched plate tension and double cantilever beam delamination benchmarks.

- **Non-Linear Monolithic**
	- Current implementation struggles with convergence
	- Energy history function relaxed during each step
	- Further work needed
- **Materials Function Benchmark**
	- Script used to optimise the calculation of the modified stiffness tensor and elastic free energy
- **Modified Newton Inertia Correction**
	- Script used to develop the inertia correction function that ensures the Jacobian is positive definite
	- This has been implemented in the nls.py file — this needs to be moved into the user's _site-packages/sfepy/solvers_ folder and the standard Newton non-linear solver called from a problem definition file

## Usage

Mesh files generated using [Gmsh](https://gitlab.onelab.info/gmsh/gmsh) and in the case of Sfepy, converted using the built in instance of [meshio](https://github.com/nschloe/meshio) by calling "sfepy-convert mesh.msh mesh.vtk".

Julia scripts may be called directly from the Julia Repl, or from the command line using "julia script". Sfepy scripts must be called using "sfepy-run script".

## References

[^1]: C. Miehe, M. Hofacker, and F. Welschinger, "[A phase field model for rate-independant crack propagation: Robust algorithmic implementation based on operator splits,](https://www.sciencedirect.com/science/article/pii/S0045782510001283)" Computer Methods in Applied Mechanics and Engineering, vol. 199, no. 45-48, pp. 2765–2778, 2010.

[^2]: M. M. Rahaman, “[An open-source implementation of a phase-field model for brittle fracture using gridap in julia,](https://journals.sagepub.com/doi/full/10.1177/10812865211071088)” Mathematics and Mechanics of Solids, vol. 27, no. 11, pp. 2404–2427, 2022.
