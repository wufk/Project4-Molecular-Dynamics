CUDA Molecular Dynamics
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

*  Fengkai Wu
*  Tested on: Windows 7, i7-6700 @ 3.40GHz 16GB, Quadro K620 4095MB (Moore 100C Lab)

## Introduction
Large scale computer similation is very popular now in research of materials science. The basic principle is to compute the atom to atom, or molecular to molecular interactions based on a potential function, and calucalte the dynamics in a small time step, maping the evolution of the microscopic structure. Therefore, molecular dyanmics can serve as a tool of predicting future materials design. It can also be implemented as a way to validate hypothesis.

Since materials are made of millions or trillions of atoms, this many-body simulation could take large time of time. As a result, it is significant to use scienfic computing methods to reduce the time of calculation.

This project is a simple demostration to show that molecular dynamic simulation can be accelerated by deploying on multi-core processors.

## Methodology
The basic idea of molecular dynamics is simple. According the Newton's law, 
![newton](https://github.com/wufk/Project4-Molecular-Dynamics/blob/master/images/CodeCogsEqn.png)
where v is velocity, t is time, m is mass and F is velocity.

