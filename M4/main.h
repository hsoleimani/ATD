/*
 * main.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "M4.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include "opt.h"

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
const gsl_rng_type * T;
gsl_rng * r;
double CONVERGED;
int MAXITER;
int NUMC;
int MCSIZE;
int NUMINIT;
double *gt;
int NTOP;
int update_all;
int freeze_iter;

M4_ss * new_M4_ss(M4_model* model);
M4_model* new_M4_model(int ntopics, int nterms, int ngroups, int ngenres);
M4_var * new_M4_var(M4_model* model, int nmax, int ndgmax);
void mstep(M4_corpus* corpus, M4_model* model, M4_var* var, M4_ss* ss,
		M4_alphaopt * alphaopt, gsl_vector* x, gsl_vector* x2);
void train(char* dataset, char* group_file, int ntopics, int ngroups, int ngenres,
		char* start, char* dir, char* model_name);
void test(char* dataset, char* group_file, int ngroups, char* model_name, char* dir);

int main(int argc, char* argv[]);
void write_M4_model(M4_model * model, char * root, M4_corpus * corpus);
void corpus_initialize_model(M4_model* model, M4_corpus* corpus, M4_ss* ss);
int max_corpus_length(M4_corpus* c);
M4_corpus* read_data(const char* data_filename, const char* group_filename, int ngroups);

double group_inference(M4_corpus* corpus, M4_model* model, M4_var* var, M4_ss* ss, int g, int test);

M4_model* load_model(char* model_root, int ndocs);
void read_time(M4_corpus* corpus, char * filename);

void random_initialize_model(M4_model * model, M4_corpus* corpus, M4_ss* ss);
void write_word_assignment(M4_corpus* c,char * filename, M4_model* model);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
