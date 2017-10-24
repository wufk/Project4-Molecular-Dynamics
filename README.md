CUDA Molecular Dynamics
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

*  Fengkai Wu
*  Tested on: Windows 7, i7-6700 @ 3.40GHz 16GB, Quadro K620 4095MB (Moore 100C Lab)

## Introduction
Large scale computer similation is very popular now in research of materials science. The basic principle is to compute the atom to atom, or molecular to molecular interactions based on a potential function, and calucalte the dynamics in a small time step, maping the evolution of the microscopic structure. Therefore, molecular dyanmics can serve as a tool of predicting future materials design. It can also be implemented as a way to validate hypothesis.

Since materials are made of millions or trillions of atoms, this many-body simulation could take large time of time. As a result, it is significant to use scienfic computing methods to reduce the time of calculation.

This project simulates the heating/melting process of metals, a simple demostration to show that molecular dynamic simulation can be accelerated by deploying on multi-core processors.

## Methodology

### Dynamics
The basic idea of molecular dynamics is simple. According the Newton's law, 

![newton](https://github.com/wufk/Project4-Molecular-Dynamics/blob/master/images/CodeCogsEqn.png)

where v is velocity, t is time, m is mass and F is velocity. The displacement can be updated using calculus:

![xv](https://github.com/wufk/Project4-Molecular-Dynamics/blob/master/images/xv.png)

In practice, Verlet Integration is commonly used to integrate Newton's equations. This algorithm is also often used in video games. The standard implementation of a velocity Verlet algorithm is as follows:

1. ![verlet1](https://github.com/wufk/Project4-Molecular-Dynamics/blob/master/images/verlet1.png)

2. ![verlet2](https://github.com/wufk/Project4-Molecular-Dynamics/blob/master/images/verlet2.png)

3. Calculate the acceleration.

4. ![verlet3](https://github.com/wufk/Project4-Molecular-Dynamics/blob/master/images/verlet3.png)

The next step is to calculate the acceleration.

### Potential FUnction

The potential function is the key to the simulation. Researchers are devloping  more and more accurate potential functions, or emprical potential data for simluation. In this project, Lennard-Jones potential is used. The potential is described as follows:

![LJ](https://github.com/wufk/Project4-Molecular-Dynamics/blob/master/images/LJ.png)

r is the distance between two body. Sigam and Epsilon are constants which can be fitted by experiments. Then we can calculate force using

![forpo](https://github.com/wufk/Project4-Molecular-Dynamics/blob/master/images/forpo.png)

### Longevin thermostat

According to thermodynamics, the temperature is the macroscopic expression of the motion of particles, i.e. the velocity of atoms. The temperature can be calculated by the summing over the squared velocity of all atoms. 

To simulate a melting process, temperature is maintained at each each temperature interval. We use Langevin thermostat to simluate the process, which is simply adding a random force to Newton's equation when computing forces, denoting a friction between atoms.

### Analysis

The algorithm can be easily tranformed into CPU code. But there are many calculation that we can parallelize. 

When initializing the lattice, we can simply call a kernel to assign initial positions and velocities. To "drag" the temperature of the system to the initial temperature, kinetic energy is needed to calculated. This can also be done by GPU reduciton. 

Stepping over the simluation, we can see that the Verlet Integration can also be parallelized. The updation of velocity and displacement can be easily calculated by calling a simpole kernel. Therefore, the main challenge is how to compute the interations of atoms.


