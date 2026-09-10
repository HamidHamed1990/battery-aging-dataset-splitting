# battery-aging-dataset-splitting
Reproducible Python algorithms for restructuring and segmenting battery cycling aging datasets.
# Battery Aging Dataset Splitting

This repository contains reproducible Python implementations for restructuring
and segmenting battery cycling aging datasets used in published research.

Each dataset-specific directory contains:

- a publication-ready Python preprocessing module;
- a scientific description of the segmentation algorithm;
- expected input and output structures;
- configurable numerical parameters;
- command-line usage examples;
- methodological limitations and assumptions.

## Repository structure

- `datasets/zhang/`: Excel restructuring and current-extremum segmentation
- `datasets/current-threshold-gradient/`: polarity and current-gradient segmentation
- `datasets/lu/`: physics-informed CC/CV phase classification
- `datasets/mohtat/`: protocol-aware charge, discharge, and impedance segmentation
- `datasets/matlab-cycle-patterns/`: cycle reconstruction from MATLAB operation sequences
- `datasets/v73-battery-batches/`: MATLAB v7.3 extraction and plateau-based segmentation

## Data availability

The original research datasets are not distributed in this repository.
Users must obtain each dataset from its official source and comply with its
license and terms of use.

## Installation

Create a Python environment and install the required packages:

```bash
python -m venv .venv
