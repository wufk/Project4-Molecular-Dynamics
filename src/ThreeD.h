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
  ThreeD(int);
  ~ThreeD();
  void MDLoop(int);
  void init();
  //void input();
//  void CNP();
  int N;

private:
	float A, B, C, D;
	float dcut2, dcut;
  float m, dt, hdt, sigma, epsilon, t, tau, inv_m;//hdt = half dt
  float kB, ke, pe, T, T0, Tt;
  int istep, ifreq, nunit, nstep, nall, s_tp, Crate;//s_tp = structure
  void force(float);
  //void output();
  void cell_init(int);
  FILE *f1, *f2; 
  Memory *M;
  RanPark *rnd;
  string *cfg;
  float **x, **f, **v;
  float ncell[Dim];
  float L[Dim];
  float hL[Dim];
  float a0[Dim];
  char Atom[2];
};

#endif
