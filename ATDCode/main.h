
#ifndef MAIN_H_
#define MAIN_H_

#include "APD.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "cokus.h"

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)

double CONVERGED;
int MAXITER;
double NU;
int AutoBreak;

int main(int argc, char* argv[]);
void atd(char* dataset, char* dir, char* model_name, char* titlefile, int lmax, char* vfile);
double CompScomLkh(atd_corpus* corpus, atd_model* model, atd_ss* ss, atd_var* var);
void nullmodel(atd_corpus* corpus, atd_model* model, atd_ss* ss, atd_var* var);
void update_null_theta(double* theta0, double* theta1, double* sstheta, int* v0,
		double* phi, atd_model* model, document* doc);
void update_alt_theta(double* theta0, double* theta1, double* sstheta, int* v0,
		double* phi, atd_model* model, document* doc);
void btstp2(char* dataset, char* dir, char* model_name, char* thetafile, int Ssize,
		int BSSize, char* doclenfile);
double LearnUnderAltModel(atd_corpus* corpus, atd_model* model, atd_ss* ss, atd_var* var);
double TrainAltModel(atd_corpus* corpus, atd_model* model, atd_ss* ss, atd_var* var);
double compute_likelihood(atd_corpus* corpus, atd_var* var, atd_model* model);
void valid_nullmodel(atd_corpus* corpus, atd_model* model, atd_valid_var* var);
double SingleDocBtstp(document* doc, atd_corpus* valid_corpus, atd_valid_var* valid_var,
		atd_model* model, int d0);
atd_trmodel* new_atd_trmodel(int ntopics, int nterms, int ndocs);
atd_model* new_atd_model(int ntopics, int nterms, int ndocs, char* model_name);
atd_var * new_atd_var(int ndocs, int ntopics, int nmax, int nterms);
atd_ss * new_atd_ss(atd_model* model);
atd_valid_var * new_atd_valid_var(atd_corpus* valid_corpus, int ntopics);
int max_corpus_length(atd_corpus* c);
atd_trmodel* load_model(char* model_root, int ndocs);
atd_corpus* read_data(const char* data_filename,int ntopics);
void write_atd_model(atd_model * model, char * root, atd_corpus * corpus, atd_var* var);

#endif /* MAIN_H_ */
