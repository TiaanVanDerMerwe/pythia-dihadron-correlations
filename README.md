# PYTHIA Dihadron Correlations

A Monte Carlo framework for generating and analysing dihadron
correlations in proton--proton (pp) collisions using **PYTHIA 8.317**.

This project establishes a reliable (pp) reference for future heavy-ion
studies of medium-modified jet observables. It generates dihadron
correlation functions over a range of trigger and associated particle
transverse momenta, performs background subtraction using the Zero Yield
At Minimum (ZYAM) method, estimates statistical uncertainties using
jackknife resampling, and compares the resulting observables with
experimental measurements.

## Features

-   Event generation with **PYTHIA 8.317**
-   Inclusive dihadron correlation analysis
-   Configurable trigger and associated particle (p_T) selections
-   Two-dimensional
    ((`\Delta`{=tex}`\eta`{=tex},`\Delta`{=tex}`\phi`{=tex}))
    correlation functions
-   One-dimensional (`\Delta`{=tex}`\phi`{=tex}) projections
-   Folded and unfolded correlation analyses
-   ZYAM background subtraction
-   Statistical uncertainty estimation using Poisson statistics and
    jackknife resampling
-   Associate yield calculation for near- and away-side
-   Comparison with published STAR, ALICE, and CMS measurements
-   Publication-quality plotting scripts

```

## Requirements

-   C++17
-   PYTHIA 8.317
-   Python 3
-   NumPy
-   pandas
-   Matplotlib

## Author

Tiaan van der Merwe

MSc Physics, University of Cape Town
