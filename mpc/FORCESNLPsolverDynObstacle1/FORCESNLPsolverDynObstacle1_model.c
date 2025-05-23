/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) FORCESNLPsolverDynObstacle1_model_ ## ID
#endif

#include <math.h> 
#include "FORCESNLPsolverDynObstacle1_model.h"

#ifndef casadi_real
#define casadi_real FORCESNLPsolverDynObstacle1_float
#endif

#ifndef casadi_int
#define casadi_int solver_int32_default
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_f3 CASADI_PREFIX(f3)
#define casadi_f4 CASADI_PREFIX(f4)
#define casadi_f5 CASADI_PREFIX(f5)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_sign CASADI_PREFIX(sign)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#if 0
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

casadi_real casadi_sign(casadi_real x) { return x<0 ? -1 : x>0 ? 1 : x;}

static const casadi_int casadi_s0[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s1[30] = {26, 1, 0, 26, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[18] = {1, 10, 0, 1, 2, 2, 3, 4, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0};
static const casadi_int casadi_s4[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s5[21] = {3, 10, 0, 0, 0, 0, 3, 6, 8, 8, 8, 8, 8, 0, 1, 2, 0, 1, 2, 0, 1};
static const casadi_int casadi_s6[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s7[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s8[11] = {4, 6, 0, 0, 0, 0, 1, 2, 2, 0, 1};
static const casadi_int casadi_s9[11] = {6, 6, 0, 1, 2, 2, 2, 2, 2, 3, 4};
static const casadi_int casadi_s10[20] = {1, 10, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 0, 0, 0, 0, 0, 0, 0};

/* FORCESNLPsolverDynObstacle1_objective_0:(i0[10],i1[26])->(o0,o1[1x10,5nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=10.;
  a1=arg[0]? arg[0][4] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][5] : 0;
  a4=arg[1]? arg[1][1] : 0;
  a3=(a3-a4);
  a4=casadi_sq(a3);
  a2=(a2+a4);
  a2=(a0*a2);
  a4=1.0000000000000000e-02;
  a5=arg[0]? arg[0][0] : 0;
  a6=30.;
  a5=(a5/a6);
  a7=casadi_sq(a5);
  a7=(a4*a7);
  a2=(a2+a7);
  a7=arg[0]? arg[0][1] : 0;
  a7=(a7/a6);
  a6=casadi_sq(a7);
  a6=(a4*a6);
  a2=(a2+a6);
  a6=700.;
  a8=arg[0]? arg[0][3] : 0;
  a9=casadi_sq(a8);
  a9=(a6*a9);
  a2=(a2+a9);
  if (res[0]!=0) res[0][0]=a2;
  a2=3.3333333333333333e-02;
  a5=(a5+a5);
  a5=(a4*a5);
  a5=(a2*a5);
  if (res[1]!=0) res[1][0]=a5;
  a7=(a7+a7);
  a4=(a4*a7);
  a2=(a2*a4);
  if (res[1]!=0) res[1][1]=a2;
  a8=(a8+a8);
  a6=(a6*a8);
  if (res[1]!=0) res[1][2]=a6;
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[1]!=0) res[1][3]=a1;
  a3=(a3+a3);
  a0=(a0*a3);
  if (res[1]!=0) res[1][4]=a0;
  return 0;
}

int FORCESNLPsolverDynObstacle1_objective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

int FORCESNLPsolverDynObstacle1_objective_0_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolverDynObstacle1_objective_0_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_objective_0_free_mem(int mem) {
}

int FORCESNLPsolverDynObstacle1_objective_0_checkout(void) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_objective_0_release(int mem) {
}

void FORCESNLPsolverDynObstacle1_objective_0_incref(void) {
}

void FORCESNLPsolverDynObstacle1_objective_0_decref(void) {
}

casadi_int FORCESNLPsolverDynObstacle1_objective_0_n_in(void) { return 2;}

casadi_int FORCESNLPsolverDynObstacle1_objective_0_n_out(void) { return 2;}

casadi_real FORCESNLPsolverDynObstacle1_objective_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_objective_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_objective_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_objective_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_objective_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    default: return 0;
  }
}

int FORCESNLPsolverDynObstacle1_objective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolverDynObstacle1_inequalities_0:(i0[10],i1[26])->(o0[3],o1[3x10,8nz]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][4] : 0;
  a1=arg[1]? arg[1][6] : 0;
  a1=(a0-a1);
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][5] : 0;
  a4=arg[1]? arg[1][7] : 0;
  a4=(a3-a4);
  a5=casadi_sq(a4);
  a2=(a2+a5);
  a2=sqrt(a2);
  a5=arg[0]? arg[0][3] : 0;
  a6=(a2+a5);
  if (res[0]!=0) res[0][0]=a6;
  a6=arg[1]? arg[1][16] : 0;
  a6=(a0-a6);
  a7=fabs(a6);
  a8=arg[1]? arg[1][23] : 0;
  a9=2.9999999999999999e-02;
  a8=(a8+a9);
  a7=(a7/a8);
  a9=6.;
  a10=(a9*a7);
  a10=exp(a10);
  a11=(a7*a10);
  a12=arg[1]? arg[1][17] : 0;
  a3=(a3-a12);
  a12=fabs(a3);
  a13=arg[1]? arg[1][24] : 0;
  a14=4.8000000000000001e-02;
  a13=(a13+a14);
  a12=(a12/a13);
  a14=(a9*a12);
  a14=exp(a14);
  a15=(a12*a14);
  a11=(a11+a15);
  a15=(a9*a7);
  a15=exp(a15);
  a16=(a9*a12);
  a16=exp(a16);
  a17=(a15+a16);
  a11=(a11/a17);
  a18=(a11+a5);
  if (res[0]!=0) res[0][1]=a18;
  a0=(a0+a5);
  if (res[0]!=0) res[0][2]=a0;
  a0=1.;
  if (res[1]!=0) res[1][0]=a0;
  if (res[1]!=0) res[1][1]=a0;
  if (res[1]!=0) res[1][2]=a0;
  a1=(a1/a2);
  if (res[1]!=0) res[1][3]=a1;
  a6=casadi_sign(a6);
  a6=(a6/a8);
  a8=(a10*a6);
  a1=(a9*a6);
  a10=(a10*a1);
  a7=(a7*a10);
  a8=(a8+a7);
  a8=(a8/a17);
  a11=(a11/a17);
  a6=(a9*a6);
  a15=(a15*a6);
  a15=(a11*a15);
  a8=(a8-a15);
  if (res[1]!=0) res[1][4]=a8;
  if (res[1]!=0) res[1][5]=a0;
  a4=(a4/a2);
  if (res[1]!=0) res[1][6]=a4;
  a3=casadi_sign(a3);
  a3=(a3/a13);
  a13=(a14*a3);
  a4=(a9*a3);
  a14=(a14*a4);
  a12=(a12*a14);
  a13=(a13+a12);
  a13=(a13/a17);
  a9=(a9*a3);
  a16=(a16*a9);
  a11=(a11*a16);
  a13=(a13-a11);
  if (res[1]!=0) res[1][7]=a13;
  return 0;
}

int FORCESNLPsolverDynObstacle1_inequalities_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f1(arg, res, iw, w, mem);
}

int FORCESNLPsolverDynObstacle1_inequalities_0_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolverDynObstacle1_inequalities_0_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_inequalities_0_free_mem(int mem) {
}

int FORCESNLPsolverDynObstacle1_inequalities_0_checkout(void) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_inequalities_0_release(int mem) {
}

void FORCESNLPsolverDynObstacle1_inequalities_0_incref(void) {
}

void FORCESNLPsolverDynObstacle1_inequalities_0_decref(void) {
}

casadi_int FORCESNLPsolverDynObstacle1_inequalities_0_n_in(void) { return 2;}

casadi_int FORCESNLPsolverDynObstacle1_inequalities_0_n_out(void) { return 2;}

casadi_real FORCESNLPsolverDynObstacle1_inequalities_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_inequalities_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_inequalities_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_inequalities_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_inequalities_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    default: return 0;
  }
}

int FORCESNLPsolverDynObstacle1_inequalities_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolverDynObstacle1_cdyn_0:(i0[6],i1[4],i2[26])->(o0[6],o1[4x6,2nz],o2[6x6,2nz]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1;
  a0=arg[0]? arg[0][3] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0]? arg[0][4] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a0=0.;
  if (res[0]!=0) res[0][2]=a0;
  a1=arg[1]? arg[1][0] : 0;
  if (res[0]!=0) res[0][3]=a1;
  a1=arg[1]? arg[1][1] : 0;
  if (res[0]!=0) res[0][4]=a1;
  if (res[0]!=0) res[0][5]=a0;
  a0=1.;
  if (res[1]!=0) res[1][0]=a0;
  if (res[1]!=0) res[1][1]=a0;
  if (res[2]!=0) res[2][0]=a0;
  if (res[2]!=0) res[2][1]=a0;
  return 0;
}

int FORCESNLPsolverDynObstacle1_cdyn_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f2(arg, res, iw, w, mem);
}

int FORCESNLPsolverDynObstacle1_cdyn_0_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolverDynObstacle1_cdyn_0_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_cdyn_0_free_mem(int mem) {
}

int FORCESNLPsolverDynObstacle1_cdyn_0_checkout(void) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_cdyn_0_release(int mem) {
}

void FORCESNLPsolverDynObstacle1_cdyn_0_incref(void) {
}

void FORCESNLPsolverDynObstacle1_cdyn_0_decref(void) {
}

casadi_int FORCESNLPsolverDynObstacle1_cdyn_0_n_in(void) { return 3;}

casadi_int FORCESNLPsolverDynObstacle1_cdyn_0_n_out(void) { return 3;}

casadi_real FORCESNLPsolverDynObstacle1_cdyn_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_cdyn_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_cdyn_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_cdyn_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s6;
    case 1: return casadi_s7;
    case 2: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_cdyn_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s6;
    case 1: return casadi_s8;
    case 2: return casadi_s9;
    default: return 0;
  }
}

int FORCESNLPsolverDynObstacle1_cdyn_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolverDynObstacle1_cdyn_0rd_0:(i0[6],i1[4],i2[26])->(o0[6]) */
static int casadi_f3(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1;
  a0=arg[0]? arg[0][3] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0]? arg[0][4] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a0=0.;
  if (res[0]!=0) res[0][2]=a0;
  a1=arg[1]? arg[1][0] : 0;
  if (res[0]!=0) res[0][3]=a1;
  a1=arg[1]? arg[1][1] : 0;
  if (res[0]!=0) res[0][4]=a1;
  if (res[0]!=0) res[0][5]=a0;
  return 0;
}

int FORCESNLPsolverDynObstacle1_cdyn_0rd_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f3(arg, res, iw, w, mem);
}

int FORCESNLPsolverDynObstacle1_cdyn_0rd_0_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolverDynObstacle1_cdyn_0rd_0_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_cdyn_0rd_0_free_mem(int mem) {
}

int FORCESNLPsolverDynObstacle1_cdyn_0rd_0_checkout(void) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_cdyn_0rd_0_release(int mem) {
}

void FORCESNLPsolverDynObstacle1_cdyn_0rd_0_incref(void) {
}

void FORCESNLPsolverDynObstacle1_cdyn_0rd_0_decref(void) {
}

casadi_int FORCESNLPsolverDynObstacle1_cdyn_0rd_0_n_in(void) { return 3;}

casadi_int FORCESNLPsolverDynObstacle1_cdyn_0rd_0_n_out(void) { return 1;}

casadi_real FORCESNLPsolverDynObstacle1_cdyn_0rd_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_cdyn_0rd_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_cdyn_0rd_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_cdyn_0rd_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s6;
    case 1: return casadi_s7;
    case 2: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_cdyn_0rd_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s6;
    default: return 0;
  }
}

int FORCESNLPsolverDynObstacle1_cdyn_0rd_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolverDynObstacle1_objective_1:(i0[10],i1[26])->(o0,o1[1x10,7nz]) */
static int casadi_f4(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=10.;
  a1=arg[0]? arg[0][4] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][5] : 0;
  a4=arg[1]? arg[1][1] : 0;
  a3=(a3-a4);
  a4=casadi_sq(a3);
  a2=(a2+a4);
  a2=(a0*a2);
  a4=1.0000000000000000e-02;
  a5=arg[0]? arg[0][0] : 0;
  a6=30.;
  a5=(a5/a6);
  a7=casadi_sq(a5);
  a7=(a4*a7);
  a2=(a2+a7);
  a7=arg[0]? arg[0][1] : 0;
  a7=(a7/a6);
  a6=casadi_sq(a7);
  a6=(a4*a6);
  a2=(a2+a6);
  a6=700.;
  a8=arg[0]? arg[0][3] : 0;
  a9=casadi_sq(a8);
  a9=(a6*a9);
  a2=(a2+a9);
  a9=1.0000000000000001e-01;
  a10=arg[0]? arg[0][7] : 0;
  a11=casadi_sq(a10);
  a11=(a9*a11);
  a2=(a2+a11);
  a11=arg[0]? arg[0][8] : 0;
  a12=casadi_sq(a11);
  a12=(a9*a12);
  a2=(a2+a12);
  if (res[0]!=0) res[0][0]=a2;
  a2=3.3333333333333333e-02;
  a5=(a5+a5);
  a5=(a4*a5);
  a5=(a2*a5);
  if (res[1]!=0) res[1][0]=a5;
  a7=(a7+a7);
  a4=(a4*a7);
  a2=(a2*a4);
  if (res[1]!=0) res[1][1]=a2;
  a8=(a8+a8);
  a6=(a6*a8);
  if (res[1]!=0) res[1][2]=a6;
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[1]!=0) res[1][3]=a1;
  a3=(a3+a3);
  a0=(a0*a3);
  if (res[1]!=0) res[1][4]=a0;
  a10=(a10+a10);
  a10=(a9*a10);
  if (res[1]!=0) res[1][5]=a10;
  a11=(a11+a11);
  a9=(a9*a11);
  if (res[1]!=0) res[1][6]=a9;
  return 0;
}

int FORCESNLPsolverDynObstacle1_objective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f4(arg, res, iw, w, mem);
}

int FORCESNLPsolverDynObstacle1_objective_1_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolverDynObstacle1_objective_1_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_objective_1_free_mem(int mem) {
}

int FORCESNLPsolverDynObstacle1_objective_1_checkout(void) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_objective_1_release(int mem) {
}

void FORCESNLPsolverDynObstacle1_objective_1_incref(void) {
}

void FORCESNLPsolverDynObstacle1_objective_1_decref(void) {
}

casadi_int FORCESNLPsolverDynObstacle1_objective_1_n_in(void) { return 2;}

casadi_int FORCESNLPsolverDynObstacle1_objective_1_n_out(void) { return 2;}

casadi_real FORCESNLPsolverDynObstacle1_objective_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_objective_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_objective_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_objective_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_objective_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s10;
    default: return 0;
  }
}

int FORCESNLPsolverDynObstacle1_objective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolverDynObstacle1_inequalities_1:(i0[10],i1[26])->(o0[3],o1[3x10,8nz]) */
static int casadi_f5(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][4] : 0;
  a1=arg[1]? arg[1][6] : 0;
  a1=(a0-a1);
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][5] : 0;
  a4=arg[1]? arg[1][7] : 0;
  a4=(a3-a4);
  a5=casadi_sq(a4);
  a2=(a2+a5);
  a2=sqrt(a2);
  a5=arg[0]? arg[0][3] : 0;
  a6=(a2+a5);
  if (res[0]!=0) res[0][0]=a6;
  a6=arg[1]? arg[1][16] : 0;
  a6=(a0-a6);
  a7=fabs(a6);
  a8=arg[1]? arg[1][23] : 0;
  a9=2.9999999999999999e-02;
  a8=(a8+a9);
  a7=(a7/a8);
  a9=6.;
  a10=(a9*a7);
  a10=exp(a10);
  a11=(a7*a10);
  a12=arg[1]? arg[1][17] : 0;
  a3=(a3-a12);
  a12=fabs(a3);
  a13=arg[1]? arg[1][24] : 0;
  a14=4.8000000000000001e-02;
  a13=(a13+a14);
  a12=(a12/a13);
  a14=(a9*a12);
  a14=exp(a14);
  a15=(a12*a14);
  a11=(a11+a15);
  a15=(a9*a7);
  a15=exp(a15);
  a16=(a9*a12);
  a16=exp(a16);
  a17=(a15+a16);
  a11=(a11/a17);
  a18=(a11+a5);
  if (res[0]!=0) res[0][1]=a18;
  a0=(a0+a5);
  if (res[0]!=0) res[0][2]=a0;
  a0=1.;
  if (res[1]!=0) res[1][0]=a0;
  if (res[1]!=0) res[1][1]=a0;
  if (res[1]!=0) res[1][2]=a0;
  a1=(a1/a2);
  if (res[1]!=0) res[1][3]=a1;
  a6=casadi_sign(a6);
  a6=(a6/a8);
  a8=(a10*a6);
  a1=(a9*a6);
  a10=(a10*a1);
  a7=(a7*a10);
  a8=(a8+a7);
  a8=(a8/a17);
  a11=(a11/a17);
  a6=(a9*a6);
  a15=(a15*a6);
  a15=(a11*a15);
  a8=(a8-a15);
  if (res[1]!=0) res[1][4]=a8;
  if (res[1]!=0) res[1][5]=a0;
  a4=(a4/a2);
  if (res[1]!=0) res[1][6]=a4;
  a3=casadi_sign(a3);
  a3=(a3/a13);
  a13=(a14*a3);
  a4=(a9*a3);
  a14=(a14*a4);
  a12=(a12*a14);
  a13=(a13+a12);
  a13=(a13/a17);
  a9=(a9*a3);
  a16=(a16*a9);
  a11=(a11*a16);
  a13=(a13-a11);
  if (res[1]!=0) res[1][7]=a13;
  return 0;
}

int FORCESNLPsolverDynObstacle1_inequalities_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f5(arg, res, iw, w, mem);
}

int FORCESNLPsolverDynObstacle1_inequalities_1_alloc_mem(void) {
  return 0;
}

int FORCESNLPsolverDynObstacle1_inequalities_1_init_mem(int mem) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_inequalities_1_free_mem(int mem) {
}

int FORCESNLPsolverDynObstacle1_inequalities_1_checkout(void) {
  return 0;
}

void FORCESNLPsolverDynObstacle1_inequalities_1_release(int mem) {
}

void FORCESNLPsolverDynObstacle1_inequalities_1_incref(void) {
}

void FORCESNLPsolverDynObstacle1_inequalities_1_decref(void) {
}

casadi_int FORCESNLPsolverDynObstacle1_inequalities_1_n_in(void) { return 2;}

casadi_int FORCESNLPsolverDynObstacle1_inequalities_1_n_out(void) { return 2;}

casadi_real FORCESNLPsolverDynObstacle1_inequalities_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_inequalities_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FORCESNLPsolverDynObstacle1_inequalities_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_inequalities_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FORCESNLPsolverDynObstacle1_inequalities_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    default: return 0;
  }
}

int FORCESNLPsolverDynObstacle1_inequalities_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
