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
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>


#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-30
#define PI 3.14159265359
#define max(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a > _b ? _a : _b; })



typedef struct
{
    int* words;
    int* counts;
    int length;
    int total;
} document;


typedef struct
{
    document* docs;
    int nterms;
    int ndocs;
    int ngroups;
    int** groups;
    int* ngdocs;
} M4_corpus;


typedef struct M4_model
{
    int ntopics;
    int ngenres;
    int nterms;
    int ndocs;
    int ngroups;
    double** logbeta;
    double** alpha;
    double* pi;
} M4_model;


typedef struct M4_var
{
    double**** phi;
    double*** sumphi;
    double*** gamma;
    double** sumgamma;
    double* oldphi;
    double* mu;
    double* oldmu;
    double* Qt;
    double*** logtheta;
} M4_var;


typedef struct M4_ss
{
    double** beta;
    double* sumbeta;
    double* pi;
    double*** alpha;
} M4_ss;

typedef struct M4_alphaopt
{
    double** mu;
    int* ngdocs;
    //int** groups;
    double*** alphass;
    double* sumalpha;
    double* sumlgalpha;
    double** lgalpha;
    double** alpha;
    double* grad;
    int ngroups;
    int ngenres;
    int ntopics;
} M4_alphaopt;





#endif /* SPARSELDA_H_ */
