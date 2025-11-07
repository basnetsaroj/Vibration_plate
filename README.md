# Free Vibration Analysis of a Simplified Compressor Blade with NACA 2412 Thickness Distribution

This repository contains the implementation and results of an independent research project on the vibration characteristics of a compressor blade modeled as a variable-thickness rectangular plate.

## Overview
A semi-analytical model was developed using **Reissner–Mindlin plate theory** and the **Rayleigh–Ritz method** to analyze free vibration behavior of a rectangular plate with a **NACA 2412** thickness profile and **CFFF (Clamped–Free–Free–Free)** boundary condition.

The formulation was implemented in **Python**, and results were validated with **ANSYS FEM**, showing less than **3 % deviation** in the first six natural frequencies.

## Repository Structure
- `Calculation Code/` – Python scripts for Rayleigh–Ritz formulation, stiffness/mass matrix assembly, and eigenvalue solution  
- `ANSYS/` – FEM model setup and mode shape images  
- `Data calc.xlsx` – Numerical data and computed frequency comparison  
- `Report: Free Vibration Analysis of simplified blade.pdf` – Full technical report  
- `README.md` – Project summary  

## Current Work
The model is being extended under the supervision of Asst. Prof. Bikki Chhantyal (Thapathali Campus) to include **taper, twist, and rotational effects** (centrifugal stiffening and gyroscopic coupling) for more realistic compressor-blade dynamics.

## Citation
If you refer to this work, please cite as:
> Basnet, S. (2025). *Free Vibration Analysis of a Simplified Compressor Blade with NACA 2412 Thickness Distribution (Independent Research)*.
