# Phase-Field Method for Delamination Using Gridap and SfePy

## Current Status

### Julia & Gridap

- Linear segregated: working
- Non-linear staggered: working
- Non-linear coupled: not working
  - Fracture response is "soft" and doesn't fully fracture across plate

### Python & SfePy

- Linear segregated: in development
  - Elastic energy field is "out-of-date" before being used in displacement equation
  - Elastic energy term is split into two terms
  - Mesh is a rectangular test mesh with triangular elements

### Problems

- Single edge notched plate
  - Gridap & SfePy
- Double cantilever beam
  - TBC

## ToDo

### Julia & Gridap

- Implement energy based active step control
- Try and fix non-linear coupled problem
- Implement other problems
  - Double cantilever beam
  - Interface fracture

### Python & SfePy

- Fix issues with linear segregated
- Implement proper mesh using SfePy mesh conversion tools
- Dt is fixed and needs to either be adaptive, or displacement based (add to step hook)
