# Project Overview

## Mission
Build a modern, extensible neutron resonance imaging toolkit that supports VENUS and MARS beamline workflows while remaining broadly useful to the neutron imaging community.

## Primary users
- Regular users: known isotopes, unknown abundances, need spatially resolved abundance maps.
- Advanced users: mostly known isotopes, partial unknowns, need joint identification and quantification.
- Nuclear physics users: measure cross-sections for specific isotopes rather than relying on ENDF.

## Core workflows
- Forward model for transmission and related corrections.
- Parameter estimation for abundance, thickness, and cross-section inference.
- NeXus/HDF5 I/O and reproducible analysis pipelines.

## Non-goals (initial)
- Full SAMMY feature parity.
- Facility-wide data management systems.
- General-purpose tomography reconstruction.

## Guiding principles
- Physics-first: make models explicit and testable.
- Modularity: separate physics, I/O, and optimization layers.
- Reproducibility: stable inputs, clear provenance, and metadata.
