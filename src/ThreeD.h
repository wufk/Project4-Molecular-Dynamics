#ifndef ThreeD_H
#define ThreeD_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"memory.h"
#include"random.h"
using namespace std;
#define Dim 3
#define MAX_LEN 512

class ThreeD{
public:
  ThreeD();
  ~ThreeD();
  void MDLoop(int);
  void init();
  void input();
//  void CNP();
  int N;

private:
  double m, dt, hdt, sigma, epsilon, t, tau, inv_m;//hdt = half dt
  double kB, ke, pe, T, T0, Tt;
  int istep, ifreq, nunit, nstep, nall, s_tp, Crate;//s_tp = structure
  void force(int);
  void output();
  void cell_init(int);
  FILE *f1, *f2; 
  Memory *M;
  RanPark *rnd;
  string *cfg;
  double **x, **f, **v;
  double ncell[Dim];
  double L[Dim];
  double a0[Dim];
  char Atom[2];
};

#endif
