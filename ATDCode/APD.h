/*
 * sparseLDA.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef SPARSELDA_H_
#define SPARSELDA_H_

//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
#include <time.h>
//#include "multimin.h"
//#define GET(x,i) gsl_vector_get(x,i)
//#define SET(x,i,y) gsl_vector_set(x,i,y)

#define NUM_INIT 10
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-100
//#define NU  1e-5
#define PI 3.14159265359
 #define max(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a > _b ? _a : _b; })
int LBFGS_MAXITER;
int LBFGS_ORDER;
int SEARCH;

typedef struct
{
    int* words;
    int* counts;
    double* normcnts;
    //double** phi;
    double tpcont;
    int length;
    int total;
    char title[400];
} document;

typedef struct
{
    document* docs;
    int nterms;
    int ndocs;
} atd_corpus;


typedef struct atd_trmodel
{
    int m;
    int D;
    int n;
    double** beta;
    double** theta;
    int** v;
} atd_trmodel;

typedef struct atd_model
{
    int m;
    int D;
    int n;
    double** beta0;
    double** theta0;
    int** v0;
    double** beta;
    double** theta;
    int** v;
    double* shared;
    double* betahat;
    int* u;
    double zeta;
    double mu;
    double sumshared;
} atd_model;


typedef struct atd_ss
{
    double* beta;
    double sum_beta;
    double** beta0;
    double* sum_beta0;
    double** theta;
} atd_ss;


typedef struct atd_var
{
    double** phi;
    double* oldphi;
    //double* sumphi;
    double* lkh0;
    int firstdoc;
    int* sdocs;
    int* swrds;
    int* docchk;
    int* wrdchk;
    int* tempwrdchk;
    int slen;
    int swrdlen;
    double* lkh;
    double* lkhratio;
	double* temp_betass0;
	double* temp_sumbetass0;
	double X2_alt;
	double X2_null;
	double sumnd;
} atd_var;

typedef struct atd_valid_var
{
    double* phi;
    double** theta0;
    int** v0;
    double* normtheta0;
    double* theta;
    double* temp_theta;
    double* sstheta;
    int* simdocs;
    int* v;
    //int* words;
    //int* counts;
    document* doc;
    int B;
    double* pwords;
} atd_valid_var;

#endif /* SPARSELDA_H_ */
