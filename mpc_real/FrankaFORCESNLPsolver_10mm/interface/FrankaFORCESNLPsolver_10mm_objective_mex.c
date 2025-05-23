/*
FrankaFORCESNLPsolver_10mm : A fast customized optimization solver.

Copyright (C) 2013-2022 EMBOTECH AG [info@embotech.com]. All rights reserved.


This software is intended for simulation and testing purposes only. 
Use of this software for any commercial purpose is prohibited.

This program is distributed in the hope that it will be useful.
EMBOTECH makes NO WARRANTIES with respect to the use of the software 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. 

EMBOTECH shall not have any liability for any damage arising from the use
of the software.

This Agreement shall exclusively be governed by and interpreted in 
accordance with the laws of Switzerland, excluding its principles
of conflict of laws. The Courts of Zurich-City shall have exclusive 
jurisdiction in case of any dispute.

*/

#include "mex.h"
#include "math.h"
#include <string.h>
#include "../include/FrankaFORCESNLPsolver_10mm.h"
#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif

typedef FrankaFORCESNLPsolver_10mm_float solver_float;
typedef solver_int32_default solver_int;
#define NSTAGES ( 10 )
#define MAX(X, Y)  ((X) < (Y) ? (Y) : (X))

/* For compatibility with Microsoft Visual Studio 2015 */
#if _MSC_VER >= 1900
FILE _iob[3];
FILE * __cdecl __iob_func(void)
{
	_iob[0] = *stdin;
	_iob[1] = *stdout;
	_iob[2] = *stderr;
	return _iob;
}
#endif

/* copy functions */

void copyCArrayToM_FrankaFORCESNLPsolver_10mm(FrankaFORCESNLPsolver_10mm_float *src, double *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (double)*src++;
    }
}

void copyCValueToM_FrankaFORCESNLPsolver_10mm(FrankaFORCESNLPsolver_10mm_float* src, double* dest)
{
    *dest = (double)*src;
}

void copyMArrayToC_FrankaFORCESNLPsolver_10mm(double *src, FrankaFORCESNLPsolver_10mm_float *dest, solver_int32_default dim) 
{
    solver_int32_default i;
    for( i = 0; i < dim; i++ ) 
    {
        *dest++ = (FrankaFORCESNLPsolver_10mm_float) (*src++) ;
    }
}

void copyMValueToC_FrankaFORCESNLPsolver_10mm(double * src, FrankaFORCESNLPsolver_10mm_float * dest)
{
	*dest = (FrankaFORCESNLPsolver_10mm_float) *src;
}



extern solver_int32_default (FrankaFORCESNLPsolver_10mm_float *x, FrankaFORCESNLPsolver_10mm_float *y, FrankaFORCESNLPsolver_10mm_float *l, FrankaFORCESNLPsolver_10mm_float *p, FrankaFORCESNLPsolver_10mm_float *f, FrankaFORCESNLPsolver_10mm_float *nabla_f, FrankaFORCESNLPsolver_10mm_float *c, FrankaFORCESNLPsolver_10mm_float *nabla_c, FrankaFORCESNLPsolver_10mm_float *h, FrankaFORCESNLPsolver_10mm_float *nabla_h, FrankaFORCESNLPsolver_10mm_float *hess, solver_int32_default stage, solver_int32_default iteration, solver_int32_default threadID);
FrankaFORCESNLPsolver_10mm_extfunc pt2function_FrankaFORCESNLPsolver_10mm = &;


static void getDims(const solver_int stage, solver_int* nvar, solver_int* neq, solver_int* dimh, 
             solver_int* dimp, solver_int* diml, solver_int* dimu, solver_int* dimhl, solver_int* dimhu)
{
    const solver_int nvarArr[NSTAGES] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5}; 
    const solver_int neqArr[NSTAGES] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 0};
    const solver_int dimhArr[NSTAGES] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const solver_int dimpArr[NSTAGES] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
    const solver_int dimlArr[NSTAGES] = {3, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    const solver_int dimuArr[NSTAGES] = {2, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    const solver_int dimhlArr[NSTAGES] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const solver_int dimhuArr[NSTAGES] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    *nvar = nvarArr[stage];
    *neq = neqArr[stage];
    *dimh = dimhArr[stage];
    *dimp = dimpArr[stage];
    *diml = dimlArr[stage];
    *dimu = dimuArr[stage];
    *dimhl = dimhlArr[stage];
    *dimhu = dimhuArr[stage];
}

/* Checks all inputs and returns stage number (1-indexed) */
static void assignData(solver_int nrhs, const mxArray *prhs[], solver_int * const stage, solver_int * const nvar, solver_int * const neq, 
                    solver_int * const dimh, solver_int * const dimp, solver_int * const diml, solver_int * const dimu, solver_int * const dimhl, solver_int * const dimhu)
{
    mxArray *arr;

    if (nrhs > 3 || nrhs < 1)
	{
		mexErrMsgIdAndTxt("FORCESPRO:IncorrectInputs", "This function takes at least one input: z. And at most 3 inputs: z, p, stage.");
	}     

    // get stage
    *stage = (solver_int) 1;
    if (nrhs == 3)
    {
        arr = prhs[2];
        if ( !mxIsDouble(arr) )
        {
            mexErrMsgIdAndTxt("FORCESPRO:IncorrectInputs", "The third input (stage number) must be an integer.");
        }
        *stage = (solver_int) *mxGetPr(arr);
    }
    if ( *stage < 1 || (NSTAGES) < *stage )
    {
        mexErrMsgIdAndTxt("FORCESPRO:IncorrectInputs", "Stage must be between %d and %d.", 1, (NSTAGES));
    }    

    /* Get other dimensions */
    *stage -= 1; /* 0-indexed stages */
    getDims(*stage, nvar, neq, dimh, dimp, diml, dimu, dimhl, dimhu);

    /* Check that passed z and p have correct dims */  
    arr = prhs[0];
    if ( !mxIsDouble(arr) )
    {
        mexErrMsgIdAndTxt("FORCESPRO:IncorrectInputs", "The first input (z) must be a column vector.");
    }    
    if ( mxGetM(arr) != *nvar || mxGetN(arr) != 1 )
    {
        mexErrMsgIdAndTxt("FORCESPRO:IncorrectInputs", "The first input (z) must be a column vector of length %d.", *nvar);
    }
    if (nrhs > 1)
	{
        arr = prhs[1];
        if ( *dimp > 0 && mxIsEmpty(arr))
        {
            mexErrMsgIdAndTxt("FORCESPRO:IncorrectInputs", "The second input (p) must be a column vector of length %d.", *dimp);
        }   
        if ( !mxIsEmpty(arr) )
        {
            if ( !mxIsDouble(arr) )
            {
                mexErrMsgIdAndTxt("FORCESPRO:IncorrectInputs", "The second input (p) must be a column vector.");
            }    
            if ( mxGetM(arr) != *dimp || mxGetN(arr) != 1 )
            {
                mexErrMsgIdAndTxt("FORCESPRO:IncorrectInputs", "The second input (p) must be a column vector of length %d.", *dimp);
            }            
        }
	}
    else
    {
        if ( *dimp > 0 )
        {
            mexErrMsgIdAndTxt("FORCESPRO:IncorrectInputs", "Run time parameters are required as a second input for evaluating this fcn.");
        }         
    } 
}

/* THE mex-function */
void mexFunction( solver_int nlhs, mxArray *plhs[], solver_int nrhs, const mxArray *prhs[] )  
{
	mxArray *arr;
    solver_int nvar, neq, dimh, dimp, diml, dimu, dimhl, dimhu, stage, dimmul;

    // Allocate memory 
    solver_float *z, *p, *y, *l, *obj, *jacobj, *c, *jacc, *h, *jach, *hess;

	mxArray* obj_mex;
	mxArray* gradobj_mex;


    // get data
    assignData(nrhs, prhs, &stage, &nvar, &neq, &dimh, &dimp, &diml, &dimu, &dimhl, &dimhu);
    dimmul = diml+dimu+dimhl+dimhu;

    // Allocate memory 
    z = (solver_float *) malloc(sizeof(solver_float)*MAX(nvar,1));
    p = (solver_float *) malloc(sizeof(solver_float)*MAX(dimp,1));
    y = (solver_float *) malloc(sizeof(solver_float)*MAX(neq,1));
    l = (solver_float *) malloc(sizeof(solver_float)*MAX(dimmul,1));
    obj = (solver_float *) malloc(sizeof(solver_float));
    jacobj = (solver_float *) malloc(sizeof(solver_float)*MAX(nvar,1));
    c = (solver_float *) malloc(sizeof(solver_float)*MAX(neq,1));
    jacc = (solver_float *) malloc(sizeof(solver_float)*MAX(neq*nvar,1));
    h = (solver_float *) malloc(sizeof(solver_float)*MAX(dimh,1));
    jach = (solver_float *) malloc(sizeof(solver_float)*MAX(nvar*dimh,1));
    hess = (solver_float *) malloc(sizeof(solver_float)*MAX(nvar*nvar,1));

    /* Initialize all inputs */
    arr = prhs[0];
    copyMArrayToC_FrankaFORCESNLPsolver_10mm(mxGetPr(arr), z, nvar);
    if (nrhs > 1)
	{
        arr = prhs[1];
        if ( !mxIsEmpty(arr) )
        {
            copyMArrayToC_FrankaFORCESNLPsolver_10mm(mxGetPr(arr), p, dimp);
        }
	}   
    memset(y, 0, sizeof(solver_float)*neq);
    memset(l, 0, sizeof(solver_float)*dimmul);
    memset(obj, 0, sizeof(solver_float));
    memset(jacobj, 0, sizeof(solver_float)*nvar);
    memset(c, 0, sizeof(solver_float)*neq);
    memset(jacc, 0, sizeof(solver_float)*neq*nvar);
    memset(h, 0, sizeof(solver_float)*dimh);
    memset(jach, 0, sizeof(solver_float)*dimh*nvar);
    memset(hess, 0, sizeof(solver_float)*nvar*nvar);

    // Evaluate fcns and read output into mex format
	(z, y, l, p, obj, jacobj, c, jacc, h, jach, hess, stage, 0, 0);
	obj_mex = mxCreateDoubleMatrix(1, 1, mxREAL);
	gradobj_mex = mxCreateDoubleMatrix(nvar, 1, mxREAL);
	copyCArrayToM_FrankaFORCESNLPsolver_10mm(obj, mxGetPr(obj_mex), 1);
	copyCArrayToM_FrankaFORCESNLPsolver_10mm(jacobj, mxGetPr(gradobj_mex), nvar);
	plhs[0] = obj_mex;
	plhs[1] = gradobj_mex;


    // Free memory
    free(z); free(p); free(y); free(l); free(obj); free(jacobj); free(c); free(jacc); free(h); free(jach); free(hess);
}