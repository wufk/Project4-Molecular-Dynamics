#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ThreeD.h"
#include "random.h"
#include "memory.h"
#include <iostream>
using namespace std;

/* ---------------------------------------------------------------------- */
ThreeD::ThreeD()
{
  m = 1.;
  inv_m = 1./m;
  ke = pe = t = 0.0;
  istep = nstep = 0;
  nstep = 100;//loop number
  ifreq = 10;//output frequency
  s_tp = 1;//fcc
  dt = 0.0005;
  tau = 0.01 / dt;//in fact inv_tau
  hdt = 0.5 * dt;
  sigma = 1.; epsilon = 1.;
  M = NULL; 
  f1 = f2 = NULL;
  x = f = v = NULL;
  f1 = f2 = NULL;
  kB = 1.; 
  Tt = 10.5;//Ttarget
  T0 = 0.5;//init temperature
  rnd = new RanPark(1234);
  M = new Memory();
  nunit = 4;
  strcpy(Atom, "Cr");  
  for(int i = 0; i < Dim; i++){
    ncell[i] = 4.0;
    L[i] = 0.0;
    a0[i] = 1.5;
  }
 
  Crate = 20;
  N = (Tt - T0)/(Crate * dt * nstep);
  cfg = new string[MAX_LEN];


  return;
}

ThreeD::~ThreeD()
{
  if(x)M->destroy(x);
  if(f)M->destroy(f);
  if(v)M->destroy(v);
  delete rnd;
  delete M;
  delete []cfg;
  return;
}

/* ---------------------------------------------------------------------- */
void ThreeD::cell_init(int k)
{
  float **cell = nullptr;
  switch (k){
    case 1:{
    nall = nunit = 4;//fcc
    for(int i = 0; i < Dim; i++) nall *= ncell[i];
    for(int i = 0; i < Dim; i++) L[i] = float(ncell[i]) * a0[i];  
    M->create(cell, 4, 3, "cell");
    cell[0][0] = cell[0][1] = cell[0][2] = 0.0;
    cell[1][0] = 0.0        ; cell[1][1] = 0.5 * a0[1]; cell[1][2] = 0.5 * a0[2];
    cell[2][0] = 0.5 * a0[0]; cell[2][1] = 0.0        ; cell[2][2] = 0.5 * a0[2];
    cell[3][0] = 0.5 * a0[0]; cell[3][1] = 0.5 * a0[1]; cell[3][2] = 0.0        ;
    break;
    }
    
    case 2:{
    nall = nunit = 2;//bcc
    for(int i = 0; i < Dim; i++) nall *= ncell[i];
    for(int i = 0; i < Dim; i++) L[i] = float(ncell[i]) * a0[i];  
    M->create(cell, 2, 3, "cell");
    cell[0][0] = cell[0][1] = cell[0][2] = 0.;
    cell[1][0] = 0.5 * a0[0]; cell[1][1] = 0.5 * a0[1]; cell[1][2] = 0.5 * a0[2];
    break;
    } 
    
    case 3:{
    nall = nunit = 1;//simple cubic
    for(int i = 0; i < Dim; i++) nall *= ncell[i];
    for(int i = 0; i < Dim; i++) L[i] = float(ncell[i]) * a0[i];  
    M->create(cell, 1, 3, "cell");
    cell[0][0] = cell[0][1] = cell[0][2] = 0.;
    }

    default: printf("Cell initialization error!Please input number.\n");
  }
  
  M->create(x, nall, Dim, "x");
  //printf("initilizing\n");
  int ii=0 ;
  for (int ix = 0; ix < ncell[0]; ix++)
  for (int iy = 0; iy < ncell[1]; iy++)
  for (int iz = 0; iz < ncell[2]; iz++)
  for (int iu = 0; iu < nunit; iu++){
    x[ii][0] = float(ix) * a0[0] + cell[iu][0];
    x[ii][1] = float(iy) * a0[1] + cell[iu][1];
    x[ii][2] = float(iz) * a0[2] + cell[iu][2];
  //for(int k = 0; k < Dim; k++){ 
 //   x[ii][k] = float(ncell[k]) * a0[k] + cell[iu][k];
    //x[ii][0] =float(ix) * d + cell[iu][0];// + rnd->gaussian() * 0.02 * d;
  //} 
    ++ii;
  }
  
  M->destroy(cell);
  return;
}
/* ---------------------------------------------------------------------- */

void ThreeD::init()
{
  printf("init start\n"); 

  //printf("initilizing cell(fcc)\n");
  cell_init(s_tp);

  //velocity init
  M->create(v, nall, Dim, "v");
  float mon[Dim] = {0.0,0.0,0.0};//mean velocity
  for(int i = 0; i < nall; i++)
  for(int j = 0; j < Dim; j++){
    v[i][j] = rnd->uniform() - 0.5;
    mon[j] += v[i][j];
  }

  //printf("compute initial kinetic energy and temperature\n");  
  
  for(int i = 0; i < 3; i++) mon[i] /= float(nall);
  ke = 0.;
  for(int i = 0; i < nall; i++){
  for(int j = 0; j < 3; j++){
    v[i][j] -= mon[j];
    ke += v[i][j] * v[i][j];
  }
  }
  ke *= 0.5 * m;
  T = ke /(1.5 * float(nall) * kB);
  float gama = sqrt(T0/T);
  
  ke = 0.0;
  for(int i = 0; i < nall; i++)
  for(int j = 0; j < 3; j++){
    v[i][j] *= gama;
    ke += v[i][j]*v[i][j];
  }
  ke *= 0.5 * m;
  T = ke /(1.5 * float(nall) * kB);

  M->create(f, nall, Dim, "f");
  force(0);
  printf("init completed\n");

  //char tem[MAX_LEN];
  //sprintf(tem, "%f", T0);
  //string a = "../dump/cfg";
  //string b = ".xyz";
  //string c = tem;
  //c = a+c+b;
  //const char *str = c.c_str();
  //f2 = fopen(str, "w");
  //f1 = fopen("../dump/log.dat", "w");
  //output();
  //fclose(f2);fclose(f1);

  printf("ke = %f pe = %f T = %f\n",ke, pe, T);

  return;
}

/* ---------------------------------------------------------------------- */

void ThreeD::MDLoop(int k)
{
  float TT = T0;
  float Gc = (Tt - T0)/(N - 1);
  string a = "../dump/cfg";
  string b = ".xyz";
  printf("\n");
  //for(int i = 0; i < N; i++){
  //  char temp[MAX_LEN];
  //  sprintf(temp, "%f", TT);
  //  string c = temp;
  //  cfg[i] = a+c+b;
  //  TT += Gc;
  //}
  //
  //f1 = fopen("../dump/log.dat", "w");
  //fprintf(f1,"#step time ke pe\n");
  //
  //const char *cstr = cfg[k].c_str();
  //printf("%s\n", cstr);
  //f2 = fopen(cstr, "w");

  for(istep = 0; istep < nstep; istep++){
    //increase v by half
    for(int i = 0; i < nall; i++){
    for(int j = 0; j < 3; j++) v[i][j] += f[i][j] * inv_m * hdt; 
    }

    //increase x by one
    for(int i = 0; i < nall; i++){
      for(int j = 0; j < 3; j++) x[i][j] += v[i][j] * dt;
    }
   
	float TT = T0;
	float Gc = (Tt - T0) / (N - 1);
	TT = TT + k * Gc;
	pe = 0.0;
	float coef = sqrt(24. * tau * m * kB * TT / dt);//coef to calculate w
    force(k);

    //increase v by another half
    for(int i = 0; i < nall; i++){
    for(int j = 0; j < 3; j++) v[i][j] += f[i][j] * inv_m * hdt; 
    }
	
    //calculate ke
    ke = 0.;
    for(int i = 0; i < nall; i++)
    for(int k = 0; k < 3; k++){
      ke += v[i][k] * v[i][k]; 
    }
    ke *= 0.5 * m; t += dt;
    
    T = ke / (1.5 * float(nall) * kB);

   if(istep % ifreq == 0){
    printf("step %d ke %f pe %f T %f TT %f coef %f\n", istep, ke, pe, T, TT, coef);
   }
   //output();
  } 

  //fclose(f1);fclose(f2);
}

/* ----------------------------output------------------------------------------ */

//void ThreeD::output()
//{
//  fprintf(f1, "%d %lg %lg %lg %lg\n", istep, t, ke, pe, T);
//  fprintf(f2, "%d\n", nall);
//  fprintf(f2, "t = %f\n", t);
//  for(int i = 0;i < nall; i++){
//    fprintf(f2, "%s", Atom);
//    for(int j = 0; j < 3; j++) fprintf(f2, " %f",x[i][j]);
//    fprintf(f2, "\n");
//  }  
//  return;
//}


/* ----------------------------output------------------------------------------ 
void ThreeD::CNP()
{
  float Q;
  float dx[3] = {0.0,0.0,0.0};//displacement between two atoms in 3D
  float hL[3] = {0.5 * L[0], 0.5 * L[1], 0.5 * L[2]};
  float r, r2;
 
  for (int i = 0; i < nall ; i++){
  for (int j = 0; j < nall ; j++){
    //PBC
    r2 = 0.;
    for (int k = 0; k < 3; k++){
      dx[k] = x[j][k] - x[i][k];
      while (dx[k] >  hL[k]) dx[k] -= L[k];
      while (dx[k] < -hL[k]) dx[k] += L[k];
  //    r2 += dx[k] * dx[k];
    }
    r2 = dx[1]*dx[1] + dx[2]*dx[2] + dx[0]*dx[0];
 
  }
  }
}*/


/* ----------------------------Langevin therostat force calculation------------------------- 
*/
void ThreeD::force(int k)
{
  float TT = T0;
  float Gc = (Tt - T0)/(N - 1);
  TT = TT + k * Gc;
  //printf("TT: %f\n", TT);
  pe = 0.0;
  float coef = sqrt(24. * tau * m * kB * TT / dt);//coef to calculate w
  float dx[3] = {0.0,0.0,0.0};//displacement between two atoms in 3D
  float hL[3] = {0.5 * L[0], 0.5 * L[1], 0.5 * L[2]};
  float dcut = 2.5 * sigma;
  float dcut2 = dcut * dcut;
  float r, r2;

  for(int i = 0; i < nall;i++)
  for (int j = 0; j < 3;j++){
      f[i][j]=0.0;
  }
  
  float sigma3  = sigma  * sigma * sigma;
  float sigma6  = sigma3 * sigma3;
  float sigma12 = sigma6 * sigma6;
  
  float A = 48. * epsilon * sigma12;
  float B = -24. * epsilon * sigma6;//force constant
  float C = 4. * epsilon * sigma12;
  float D = -4. * epsilon * sigma6;//energy constant
  
  //force between atom ii and atom iii
  for (int i = 0;     i < nall - 1; i++){
  for (int j = i + 1; j < nall ;    j++){
    //PBC
    r2 = 0.;
    for (int k = 0; k < 3; k++){
      dx[k] = x[j][k] - x[i][k];
      while (dx[k] >  hL[k]) dx[k] -= L[k];
      while (dx[k] < -hL[k]) dx[k] += L[k];
  //    r2 += dx[k] * dx[k];
    }
    float r2 = dx[1]*dx[1] + dx[2]*dx[2] + dx[0]*dx[0];
    //printf("i = %d, j = %d, r2 = %f\n", i, j, r2); //r2, rtemp);
    float r6  = r2 * r2 *r2;
    float r12 = r6 * r6;
    r = sqrt(r2);
    if(r2 < dcut2 ){
      for(int k = 0; k < 3; k++){
        f[i][k] -= (A * 1./r12 + B * 1./r6) * dx[k] / r2;
        f[j][k] += (A * 1./r12 + B * 1./r6) * dx[k] / r2;
      }
      pe += C * 1./r12 + D * 1./r6;
    }
  }
  }
  for(int i = 0; i < nall; i++)
  for(int j = 0; j < 3; j++){
    float w = coef *(rnd->uniform() - 0.5);
    f[i][j] += - m * tau * v[i][j] + w;
  }
return;
}

/* ----------------------------------------------------- 
*/

//void ThreeD::input()
//{
//  char str[MAX_LEN];
//
//  printf("Please input file name:");
//  fgets(str,MAX_LEN,stdin);
//
//  char *fname;
//  char *ptr = strtok(str," \n\t\r");
//  fname = new char[strlen(ptr) + 1];
//  strcpy(fname,ptr);
//
//  FILE *fp = fopen(fname, "r");
//  if (fp == NULL){
//    printf("Error for reading");
//
//    delete []fname;
//    return;
//  }
//
//  while(fgets(str, MAX_LEN, fp))
//  {
//    char *name;
//    name = strtok(str, " \n\t\r");
//    char *data;
//    data = strtok(NULL, " \n\t\r");
//    if(strcmp(name,"atom")     == 0) strcpy(Atom, data);
//    if(strcmp(name,"timestep") == 0) dt = atof(data);
//    if(strcmp(name,"mass")     == 0) m = atof(data);
//    if(strcmp(name,"sigma")    == 0) sigma = atof(data);
//    if(strcmp(name,"epsilon")  == 0) epsilon = atof(data);
//    if(strcmp(name,"steps")    == 0) nstep = atoi(data);ifreq = nstep / 10;
//    if(strcmp(name,"xdim")     == 0) ncell[0] = atoi(data);
//    if(strcmp(name,"ydim")     == 0) ncell[1] = atoi(data);
//    if(strcmp(name,"zdim")     == 0) ncell[2] = atoi(data);
//    if(strcmp(name,"xa0")      == 0) a0[0] = atoi(data);
//    if(strcmp(name,"ya0")      == 0) a0[1] = atoi(data);
//    if(strcmp(name,"za0")      == 0) a0[2] = atoi(data);
//    if(strcmp(name,"initialT") == 0) T0 = atof(data);
//    if(strcmp(name,"targetT")  == 0) Tt = atof(data);
//    if(strcmp(name,"Crate")  == 0) Crate = atoi(data);
//    if(strcmp(name,"structure") == 0){
//      if(strcmp(data, "fcc") == 0) s_tp = 1;
//      if(strcmp(data, "bcc") == 0) s_tp = 2;
//      if(strcmp(data, "sc")  == 0) s_tp = 3;
//    }
//  }
//  fclose(fp);
//
//  delete []fname;
//return;
//  
//}