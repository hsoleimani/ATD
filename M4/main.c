#include "main.h"



int main(int argc, char* argv[])
{

	char task[40];
	char dir[400];
	char corpus_file[400];
	char group_file[400];
	char model_name[400];
	char init[400];
	int ntopics, ngroups, ngenres;
	long int seed;

	seed = atoi(argv[1]);

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

	printf("SEED = %ld\n", seed);
	gsl_rng_env_setup();

	MAXITER = 100;
	CONVERGED = 3e-4;
	NUMINIT = 10;

	//task = argv[1];
	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);
	strcpy(group_file,argv[4]);

    if (argc > 1)
    {
        if (strcmp(task, "train")==0)
        {
        	ntopics = atoi(argv[5]);
        	ngroups = atoi(argv[6]);
        	ngenres = atoi(argv[7]);
			strcpy(init,argv[8]);
			strcpy(dir,argv[9]);
			if ((strcmp(init,"load")==0))
				strcpy(model_name,argv[10]);
			train(corpus_file, group_file, ntopics, ngroups, ngenres, init, dir, model_name);

			gsl_rng_free (r);
            return(0);
        }
        if (strcmp(argv[2], "test")==0)
        {
        	ngroups = atoi(argv[5]);
			strcpy(model_name,argv[6]);
			strcpy(dir,argv[7]);
			test(corpus_file, group_file, ngroups, model_name, dir);

			gsl_rng_free (r);
            return(0);
        }
    }
    printf("usage : ac est <dataset> <# topics> <random/seed/model> <dir> <model_name>\n");
    printf("        ac inf <dataset> <model_name> <dir>\n");
    return(0);
}

void train(char* dataset, char* group_file, int ntopics, int ngroups, int ngenres,
		char* start, char* dir, char* model_name)
{
    FILE* lhood_fptr;
    FILE* fp;
    char string[100];
    char filename[100];
    int iteration;
	double lhood, prev_lhood, conv;
	//double c, y;
	int d, n, j, g, t, dd;
    M4_corpus* corpus;
    M4_model *model = NULL;
    M4_ss* ss = NULL;
    M4_var* var = NULL;
    time_t t1,t2;
    //FILE* fileptr;
    //float x;
    //double y;

    corpus = read_data(dataset, group_file, ngroups);

    int nmax, ndgmax;
    nmax = 0;
    for (d = 0; d < corpus->ndocs; d++){
    	if (corpus->docs[d].length > nmax)
    		nmax = corpus->docs[d].length;
    }
    ndgmax = 0;
    for (g = 0; g < ngroups; g++){
    	if (corpus->ngdocs[g] > ndgmax)
    		ndgmax = corpus->ngdocs[g];
    }


    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    if (strcmp(start, "seeded")==0){  //not updated
    	printf("seeded\n");
    	model = new_M4_model(ntopics, corpus->nterms, ngroups, ngenres);
    	model->ndocs = corpus->ndocs;
    	var = new_M4_var(model, nmax, ndgmax);
		ss = new_M4_ss(model);
		//random_initialize_model(mc, model, corpus, ss);
		//printf("%d\n",corpus->docs[747].length);
		corpus_initialize_model(model, corpus, ss);
		//corpus_initialize_model2(var, model, corpus);

    }
    else if (strcmp(start, "random")==0){
    	printf("random\n");
    	model = new_M4_model(ntopics, corpus->nterms, ngroups, ngenres);
    	model->ndocs = corpus->ndocs;
    	var = new_M4_var(model, nmax, ndgmax);
		ss = new_M4_ss(model);

		random_initialize_model(model, corpus, ss);

    }
    else if (strcmp(start, "load")==0){ //old
    	printf("load beta\n");
    	model = new_M4_model(ntopics, corpus->nterms, ngroups, ngenres);
    	model->ndocs = corpus->ndocs;
    	var = new_M4_var(model, nmax, ndgmax);
		ss = new_M4_ss(model);

		double y, sum;
		sprintf(filename, "%s.beta", model_name);
	    printf("loading %s\n", filename);
	    fp = fopen(filename, "r");
	    for (n = 0; n < corpus->nterms; n++){
			for (j = 0; j < model->ntopics; j++){
				fscanf(fp, " %lf", &y);
				model->logbeta[n][j] = y;
				//if (model->beta[n][j] == 0) model->beta[n][j] = EPS;
			}
		}
	    /*for (j = 0; j < model->ntopics; j++){
	    	sum = 0.0;
	    	for (n = 0; n < corpus->nterms; n++){
	    		sum += model->beta[n][j];
	    	}
	    	for (n = 0; n < corpus->nterms; n++){
	    		model->beta[n][j] /= sum;
	    	}
	    }*/
	    fclose(fp);

		sprintf(filename, "%s.alpha", model_name);
	    printf("loading %s\n", filename);
	    fp = fopen(filename, "r");
		for (t = 0; t < model->ngenres; t++){
			for (j = 0; j < model->ntopics; j++){
				fscanf(fp, " %lf", &y);
				model->alpha[t][j] = exp(y);
				//model->alpha[t][j] = gsl_rng_uniform(r);
			}
		}
	    fclose(fp);

		sprintf(filename, "%s.pi", model_name);
	    printf("loading %s\n", filename);
	    fp = fopen(filename, "r");
	    sum = 0;
		for (t = 0; t < model->ngenres; t++){
			fscanf(fp, " %lf", &y);
			model->pi[t] = exp(y);
			sum += model->pi[t];
		}
		for (t = 0; t < model->ngenres; t++){
			model->pi[t] /= sum;
		}
	    fclose(fp);
    }

    M4_alphaopt * alphaopt = NULL;
    alphaopt = malloc(sizeof(M4_alphaopt));
    alphaopt->ngroups = model->ngroups;
    alphaopt->ngenres = model->ngenres;
    alphaopt->ntopics = model->ntopics;
    alphaopt->mu = malloc(sizeof(double*)*model->ngroups);
    alphaopt->ngdocs = malloc(sizeof(int)*model->ngroups);
    //alphaopt->groups = malloc(sizeof(int*)*model->ngroups);
    for (g = 0; g < model->ngroups; g++){
    	alphaopt->ngdocs[g] = corpus->ngdocs[g];
    	///alphaopt->groups[g] = malloc(sizeof(int)*ndgmax);
    	//for (d = 0; d < corpus->ngdocs[g]; d++){
    	//	alphaopt->groups[g][d] = corpus->groups[g][d];
    	//}
    	alphaopt->mu[g] = malloc(sizeof(double)*model->ngenres);
    	for (t = 0; t < model->ngenres; t++){
    		alphaopt->mu[g][t] = 0.0;
    	}
    }
    alphaopt->alphass = malloc(sizeof(double**)*model->ngenres);
    alphaopt->sumalpha = malloc(sizeof(double)*model->ngenres);
    alphaopt->sumlgalpha = malloc(sizeof(double)*model->ngenres);
    alphaopt->alpha = malloc(sizeof(double*)*model->ngenres);
    alphaopt->lgalpha = malloc(sizeof(double*)*model->ngenres);
    alphaopt->grad = malloc(sizeof(double)*model->ngenres*model->ntopics);
    for (t = 0; t < model->ngenres; t++){
    	alphaopt->sumalpha[t] = 0.0;
    	alphaopt->sumlgalpha[t] = 0.0;
    	alphaopt->alphass[t] = malloc(sizeof(double*)*model->ntopics);
    	alphaopt->alpha[t] = malloc(sizeof(double)*model->ntopics);
    	alphaopt->lgalpha[t] = malloc(sizeof(double)*model->ntopics);
    	for (j = 0; j < model->ntopics; j++){
    		alphaopt->grad[t*model->ntopics+j] = 0.0;
    		alphaopt->alphass[t][j] = malloc(sizeof(double)*model->ngroups);
    		for (g = 0; g < model->ngroups; g++){
    			alphaopt->alphass[t][j][g] = 0.0;
    		}
    		alphaopt->alpha[t][j] = 0.0;
    		alphaopt->lgalpha[t][j] = 0.0;
    	}
    }

    double grouplkh;
	//init ss

    iteration = 0;
    sprintf(filename, "%s/%03d", dir,iteration);
    printf("%s\n",filename);
	write_M4_model(model, filename, corpus);

    time(&t1);
	prev_lhood = -1e100;

    gsl_vector *x;
    gsl_vector *x2;
	x = gsl_vector_alloc(model->ngenres*model->ntopics);
    x2 = gsl_vector_alloc(model->ngenres*model->ntopics);

	do{

		printf("***** VB ITERATION %d *****\n", iteration);
		lhood = 0.0;

		for (g = 0; g < ngroups; g++){
			//init var
			for (t = 0; t < ngenres; t++){
				var->mu[t] = 1.0/((double) ngenres);
				for (dd = 0; dd < corpus->ngdocs[g]; dd++){
					d = corpus->groups[g][dd];
					var->sumgamma[dd][t] = 0.0;
					for (j = 0; j < model->ntopics; j++){
						var->sumphi[j][dd][t] = (double)corpus->docs[d].total/((double)model->ntopics);
						var->gamma[j][dd][t] = model->alpha[t][j] + var->sumphi[j][dd][t];
						var->sumgamma[dd][t] += var->gamma[j][dd][t];
						for (n = 0; n < corpus->docs[d].length; n++){
							var->phi[n][j][dd][t] = 1.0/((double)model->ntopics);
						}
					}
				}
			}

			grouplkh = group_inference(corpus, model, var, ss, g, 0);
			lhood += grouplkh;

			//update mu in alphaopt
			//save gamma and mu for computing score
			for (t = 0; t < ngenres; t++){
				alphaopt->mu[g][t] = var->mu[t];
			}
		}

		//mstep
		mstep(corpus, model, var, ss, alphaopt, x, x2);

		conv = fabs(prev_lhood - lhood)/fabs(prev_lhood);
		if ((prev_lhood > lhood) && (conv > 1e-10)){
			printf("oops ... \n");
		}
		prev_lhood = lhood;

		time(&t2);

		sprintf(filename, "%s/%03d", dir,1);
		write_M4_model(model, filename, corpus);
		printf("%d %5.5e %5.5e %5ld \n",iteration, lhood, conv, (int)t2-t1);
		conv = (conv > 0) ? conv : -conv;
		fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, conv, (int)t2-t1);
		fflush(lhood_fptr);
		iteration ++;

	}while((iteration < MAXITER) && (conv > CONVERGED));
	fclose(lhood_fptr);

    sprintf(filename, "%s/final", dir);
    write_M4_model(model, filename, corpus);

}

void mstep(M4_corpus* corpus, M4_model* model, M4_var* var, M4_ss* ss,
		M4_alphaopt * alphaopt, gsl_vector* x, gsl_vector* x2){

	int j, n, t;
	double sumpi;

	//update beta
	for (j = 0; j < model->ntopics; j++){
		for (n = 0; n < model->nterms; n++){
			//model->beta[n][j] = ss->beta[n][j]/ss->sumbeta[j];
			if (ss->beta[n][j] == 0)
				model->logbeta[n][j] = -500;
			else
				model->logbeta[n][j] = log(ss->beta[n][j]) - log(ss->sumbeta[j]);
			//if (model->beta[n][j] == 0) model->beta[n][j] = EPS;
			ss->beta[n][j] = 0.0;
		}
		ss->sumbeta[j] = 0.0;
	}

	//update pi
	sumpi = 0.0;
	for (t = 0; t < model->ngenres; t++){
		model->pi[t] = ss->pi[t];
		sumpi += ss->pi[t];
		ss->pi[t] = 0.0;
	}
	for (t = 0; t < model->ngenres; t++){
		model->pi[t] /= sumpi;
		if (model->pi[t] == 0) model->pi[t] = EPS;
	}

	//update alpha
	int ind, g;
	double temp;
	for (t = 0; t < model->ngenres; t++){
		for (j = 0; j < model->ntopics; j++){
			ind = t*model->ntopics + j;
			temp = log(model->alpha[t][j]);
    		gsl_vector_set(x, ind, temp);
    		gsl_vector_set(x2, ind, temp);

    		// alphaopt->mu is updated outside of mstep func
    		for (g = 0; g < model->ngroups; g++){
    			alphaopt->alphass[t][j][g] = ss->alpha[t][j][g];
    			ss->alpha[t][j][g] = 0.0;
    		}
		}
	}
	optimize_alpha(x, (void *)alphaopt, model->ntopics*model->ngenres, x2);

	for (t = 0; t < model->ngenres; t++){
		for (j = 0; j < model->ntopics; j++){
			ind = t*model->ntopics + j;
			model->alpha[t][j] = exp(gsl_vector_get(x2, ind));
			if (model->alpha[t][j] == 0) model->alpha[t][j] = EPS;
		}
	}
}



double group_inference(M4_corpus* corpus, M4_model* model, M4_var* var, M4_ss* ss, int g, int test){

	int n, j, variter, w, t, dd, d, variter2;
	double c, phisum, temp, cphi, sumalpha;
	double varlkh, prev_varlkh, conv, musum, temp0;
	double varlkh2, prev_varlkh2, conv2;

	prev_varlkh = -1e100;
	conv = 0.0;
	variter = 0;
	do{
		varlkh = 0.0;

		// top layer -- loop over genres
		musum = 0.0;
		for (t = 0; t < model->ngenres; t++){
			if (var->mu[t] == 0){
				var->mu[t] = 101;
				continue;
			}else{
				var->Qt[t] = 0.0;
			}

			// second loop -- docs in the group
			for (dd = 0; dd < corpus->ngdocs[g]; dd++){

				d = corpus->groups[g][dd];
				prev_varlkh2 = -1e100;
				conv2 = 0.0;
				variter2 = 0;
				do{
					varlkh2 = 0.0;

					for (n = 0; n < corpus->docs[d].length; n++){
						w = corpus->docs[d].words[n];
						c = (double) corpus->docs[d].counts[n];
						phisum = 0.0;

						for (j = 0; j < model->ntopics; j++){
							var->oldphi[j] = var->phi[n][j][dd][t];
							var->phi[n][j][dd][t] = gsl_sf_psi(var->gamma[j][dd][t]) + (model->logbeta[w][j]);
							if (j > 0)
								phisum = log_sum(phisum, var->phi[n][j][dd][t]);
							else
								phisum = var->phi[n][j][dd][t];
						}

						for (j = 0; j < model->ntopics; j++){
							var->phi[n][j][dd][t] = exp(var->phi[n][j][dd][t] - phisum);

							temp = c*(var->phi[n][j][dd][t] - var->oldphi[j]);
							var->sumphi[j][dd][t] += temp;
							var->gamma[j][dd][t] += temp;
							var->sumgamma[dd][t] += temp;

							if (var->phi[n][j][dd][t] > 0){
								varlkh2 += c*var->phi[n][j][dd][t]*
										((model->logbeta[w][j]) - log(var->phi[n][j][dd][t]));
							}
						}

					} // end of loop over words in dd-th doc

					sumalpha = 0.0;
					temp0 = gsl_sf_psi(var->sumgamma[dd][t]);
					for (j = 0; j < model->ntopics; j++){
						sumalpha += model->alpha[t][j];
						temp = lgamma(var->gamma[j][dd][t]);
						varlkh2 += -(lgamma(model->alpha[t][j]) - temp);
						var->logtheta[j][dd][t] = gsl_sf_psi(var->gamma[j][dd][t]) - temp0;
					}
					varlkh2 += lgamma(sumalpha) - lgamma(var->sumgamma[dd][t]);

					conv2 = fabs(prev_varlkh2 - varlkh2)/fabs(prev_varlkh2);
					if ((prev_varlkh2 > varlkh2) && (conv2 > 1e-10)){
						printf("ooops doc %d, %lf %lf, %5.10e\n", d, varlkh2, prev_varlkh2, conv2);
					}
					prev_varlkh2 = varlkh2;
					variter2 ++;

				}while((variter2 < MAXITER) && (conv2 > CONVERGED));

				//need to update Qt based on varlkh2
				var->Qt[t] += varlkh2;

			}// end of loop over docs

			var->Qt[t] += log(model->pi[t]);
			var->mu[t] = var->Qt[t];
			if (musum == 0)
				musum = var->mu[t];
			else
				musum = log_sum(musum, var->mu[t]);

		}// end of loop over genres

		for (t = 0; t < model->ngenres; t++){
			if (var->mu[t] == 101){
				var->mu[t] = 0;
				continue;
			}
			var->mu[t] = exp(var->mu[t] - musum);
			if (var->mu[t] > 0)
				varlkh += var->mu[t]*(var->Qt[t] - log(var->mu[t]));
		}

		conv = fabs(prev_varlkh - varlkh)/fabs(prev_varlkh);
		if (prev_varlkh > varlkh){
			printf("ooops group %d, %lf %lf, %5.10e\n", g, varlkh, prev_varlkh, conv);
		}
		prev_varlkh = varlkh;
		variter ++;

	}while((variter < MAXITER) && (conv > CONVERGED));

	//update ss
	if (test != 1){
		for (t = 0; t < model->ngenres; t++){

			if (var->mu[t] == 0)
				continue;
			for (dd = 0; dd < corpus->ngdocs[g]; dd++){

				d = corpus->groups[g][dd];

				for (n = 0; n < corpus->docs[d].length; n++){
					w = corpus->docs[d].words[n];
					c = (double) corpus->docs[d].counts[n];
					for (j = 0; j < model->ntopics; j++){
						cphi = c*var->phi[n][j][dd][t]*var->mu[t];
						ss->beta[w][j] += cphi;
						ss->sumbeta[j] += cphi;
					}
				}
				for (j = 0; j < model->ntopics; j++){
					ss->alpha[t][j][g] += var->logtheta[j][dd][t];
				}
			}
			ss->pi[t] += var->mu[t];
		}

	}
	return(varlkh);

}


void test(char* dataset, char* group_file, int ngroups, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	int iteration, g, ngenres, t, dd, nterms;
	int d, n, j, ntopics;
	double lhood, grouplkh;
	//double sumt;

	M4_corpus* corpus;
	M4_model *model = NULL;
	M4_ss* ss = NULL;
	M4_var* var = NULL;
	time_t t1,t2;
	//float x;
	//double y;

	sprintf(filename, "%s.other", model_name);
	printf("loading %s\n", filename);
	fp = fopen(filename, "r");
	fscanf(fp, "num_topics %d\n", &ntopics);
	fscanf(fp, "num_terms %d\n", &nterms);
	fscanf(fp, "num_genres %d\n", &ngenres);
	fclose(fp);


	corpus = read_data(dataset, group_file, ngroups);

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    int nmax, ndgmax;
    nmax = 0;
    for (d = 0; d < corpus->ndocs; d++){
    	if (corpus->docs[d].length > nmax)
    		nmax = corpus->docs[d].length;
    }
    ndgmax = 0;
    for (g = 0; g < ngroups; g++){
    	if (corpus->ngdocs[g] > ndgmax)
    		ndgmax = corpus->ngdocs[g];
    }

	// set up the log likelihood log file

	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	model = load_model(model_name, corpus->ngroups);
	var = new_M4_var(model, nmax, ndgmax);
	ss = new_M4_ss(model);
	ngenres = model->ngenres;


    iteration = 0;
    sprintf(filename, "%s/test%03d", dir,iteration);
    printf("%s\n",filename);
	write_M4_model(model, filename, corpus);

    time(&t1);

	lhood = 0.0;


    double*** gammas;
    gammas = malloc(sizeof(double**)*corpus->ndocs);
    for (d = 0; d < corpus->ndocs; d++){
    	gammas[d] = malloc(sizeof(double*)*model->ngenres);
    	for (t = 0; t < model->ngenres; t++){
    		gammas[d][t] = malloc(sizeof(double)*model->ntopics);
    		for (j = 0; j < model->ntopics; j++){
    			gammas[d][t][j] = 0.0;
    		}
    	}
    }
    double** mu;
    mu = malloc(sizeof(double*)*model->ngroups);
    for (g = 0; g < model->ngroups; g++){
    	mu[g] = malloc(sizeof(double)*model->ngenres);
    	for (t = 0; t < model->ngenres; t++){
    		mu[g][t] = 0.0;
    	}
    }

	for (g = 0; g < ngroups; g++){
		//init var
		for (t = 0; t < ngenres; t++){
			var->mu[t] = 1/((double) ngenres);
			for (dd = 0; dd < corpus->ngdocs[g]; dd++){
				d = corpus->groups[g][dd];
				var->sumgamma[dd][t] = 0.0;
				for (j = 0; j < model->ntopics; j++){
					var->sumphi[j][dd][t] = (double)corpus->docs[d].total/((double)model->ntopics);
					var->gamma[j][dd][t] = model->alpha[t][j] + var->sumphi[j][dd][t];
					var->sumgamma[dd][t] += var->gamma[j][dd][t];
					for (n = 0; n < corpus->docs[d].length; n++){
						var->phi[n][j][dd][t] = 1/((double)model->ntopics);
					}
				}
			}
		}

		grouplkh = group_inference(corpus, model, var, ss, g, 1);
		lhood += grouplkh;

		//save gamma and mu for computing score
		for (t = 0; t < ngenres; t++){
			mu[g][t] = var->mu[t];
			for (dd = 0; dd < corpus->ngdocs[g]; dd++){
				d = corpus->groups[g][dd];
				for (j = 0; j < model->ntopics; j++){
					gammas[d][t][j] = var->gamma[j][dd][t];
				}
			}
		}
	}
	time(&t2);

	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, 0.0, (int)t2-t1);
	fflush(lhood_fptr);
	fclose(lhood_fptr);
	//*************************************

	sprintf(filename, "%s/testfinal", dir);
	write_M4_model(model, filename, corpus);
	//save gamma and mu
	sprintf(filename, "%s/testfinal.mu", dir);
	fp = fopen(filename, "w");
	for (g = 0; g < model->ngroups; g++){
		for (t = 0; t < model->ngenres; t++){
			fprintf(fp, "%5.10lf ", log(mu[g][t]));
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	for (t = 0; t < model->ngenres; t++){
		sprintf(filename, "%s/testfinal.gamma%d", dir,t);
		fp = fopen(filename, "w");
		for (d = 0; d < corpus->ndocs; d++){
			for (j = 0; j < model->ntopics; j++){
				fprintf(fp, "%5.10lf ", (gammas[d][t][j]));
			}
			fprintf(fp, "\n");
		}
		fclose(fp);

	}
}


M4_model* new_M4_model(int ntopics, int nterms, int ngroups, int ngenres)
{
	int n, j, t;

	M4_model* model = malloc(sizeof(M4_model));
    model->ntopics = ntopics;
    model->ngenres = ngenres;
    model->ngroups = ngroups;
    model->nterms = nterms;

    model->pi = malloc(sizeof(double)*ngenres);
    model->alpha = malloc(sizeof(double*)*ngenres);
    for (t = 0; t < ngenres; t++){
    	model->pi[t] = 0.0;
    	model->alpha[t] = malloc(sizeof(double)*ntopics);
        for (j = 0; j < ntopics; j++){
        	model->alpha[t][j] = 0.0;
        }
    }

    model->logbeta = malloc(sizeof(double*)*nterms);
    for (n = 0; n < nterms; n++){
    	model->logbeta[n] = malloc(sizeof(double)*ntopics);
    	for (j = 0; j < ntopics; j++){
    		model->logbeta[n][j] = 0.0;
    	}
    }

    return(model);
}


M4_var * new_M4_var(M4_model* model, int nmax, int ndgmax){

	int d, j, n, t;
	M4_var* var;

	var = malloc(sizeof(M4_var));
	var->phi = malloc(sizeof(double***)* nmax);
	for (n = 0; n < nmax; n++){
		var->phi[n] = malloc(sizeof(double**)* model->ntopics);
		for (j = 0; j < model->ntopics; j++){
			var->phi[n][j] = malloc(sizeof(double*)* ndgmax);
			for (d = 0; d < ndgmax; d++){
				var->phi[n][j][d] = malloc(sizeof(double)* model->ngenres);
				for (t = 0; t < model->ngenres; t++){
					var->phi[n][j][d][t] = 0.0;
				}
			}
		}
	}

	var->gamma = malloc(sizeof(double**)*model->ntopics);
	var->logtheta = malloc(sizeof(double**)*model->ntopics);
	var->sumphi = malloc(sizeof(double**)*model->ntopics);
	for (j = 0; j < model->ntopics; j++){
		var->gamma[j] = malloc(sizeof(double*)*ndgmax);
		var->logtheta[j] = malloc(sizeof(double*)*ndgmax);
		var->sumphi[j] = malloc(sizeof(double*)*ndgmax);
		for (d = 0; d < ndgmax; d++){
			var->gamma[j][d] = malloc(sizeof(double)*model->ngenres);
			var->logtheta[j][d] = malloc(sizeof(double)*model->ngenres);
			var->sumphi[j][d] = malloc(sizeof(double)*model->ngenres);
			for (t = 0; t < model->ngenres; t++){
				var->gamma[j][d][t] = 0.0;
				var->logtheta[j][d][t] = 0.0;
				var->sumphi[j][d][t] = 0.0;
			}
		}
	}
	var->sumgamma = malloc(sizeof(double*)*ndgmax);
	for (d = 0; d < ndgmax; d++){
		var->sumgamma[d] = malloc(sizeof(double)*model->ngenres);
		for (t = 0; t < model->ngenres; t++){
			var->sumgamma[d][t] = 0.0;
		}
	}


	var->mu = malloc(sizeof(double)*model->ngenres);
	var->oldmu = malloc(sizeof(double)*model->ngenres);
	var->Qt = malloc(sizeof(double)*model->ngenres);
	for (t = 0; t < model->ngenres; t++){
		var->mu[t] = 0.0;
		var->oldmu[t] = 0.0;
		var->Qt[t] = 0.0;
	}

	var->oldphi = malloc(sizeof(double)*model->ntopics);
	for (j = 0; j < model->ntopics; j++){
		var->oldphi[j] = 0.0;
	}

	return(var);
}


M4_ss * new_M4_ss(M4_model* model)
{
	int j, n, t, g;

	M4_ss * ss;
    ss = malloc(sizeof(M4_ss));
	ss->beta = malloc(sizeof(double*)*model->nterms);
	for (n = 0; n < model->nterms; n++){
		ss->beta[n] = malloc(sizeof(double)*model->ntopics);
		for (j = 0; j < model->ntopics; j++){
			ss->beta[n][j] = 0.0;
		}
	}
	ss->sumbeta = malloc(sizeof(double)*model->ntopics);
	for (j = 0; j < model->ntopics; j++){
		ss->sumbeta[j] = 0.0;
	}

	ss->pi = malloc(sizeof(double)*model->ngenres);
	ss->alpha = malloc(sizeof(double**)*model->ngenres);
	for (t = 0; t < model->ngenres; t++){
		ss->pi[t] = 0.0;
		ss->alpha[t] = malloc(sizeof(double*)*model->ntopics);
		for (j = 0; j < model->ntopics; j++){
			ss->alpha[t][j] = malloc(sizeof(double)*model->ngroups);
			for (g = 0; g < model->ngroups; g++){
				ss->alpha[t][j][g] = 0.0;
			}
		}
	}

    return(ss);
}



M4_corpus* read_data(const char* data_filename, const char* group_filename, int ngroups)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw, g;
	M4_corpus* c;

	printf("reading data from %s\n", data_filename);
	c = malloc(sizeof(M4_corpus));
	c->docs = 0;
	c->nterms = 0;
	c->ndocs = 0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "%10d", &length) != EOF)){
		c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
		c->docs[nd].length = length;
		c->docs[nd].total = 0;
		c->docs[nd].words = malloc(sizeof(int)*length);
		c->docs[nd].counts = malloc(sizeof(int)*length);
		for (n = 0; n < length; n++){
			fscanf(fileptr, "%10d:%10d", &word, &count);
			c->docs[nd].words[n] = word;
			c->docs[nd].counts[n] = count;
			c->docs[nd].total += count;
			if (word >= nw) { nw = word + 1; }
		}
		nd++;
	}
	fclose(fileptr);
	c->ndocs = nd;
	c->nterms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);

	printf("reading data from %s\n", group_filename);
	fileptr = fopen(group_filename, "r");
	c->ngroups = ngroups;
	c->groups = malloc(sizeof(int*)*ngroups);
	c->ngdocs = malloc(sizeof(int)*ngroups);
	for (g = 0; g < ngroups; g++){
		c->groups[g] = malloc(sizeof(int)*nd);
		c->ngdocs[g] = 0;
		for (n = 0; n < nd; n++)
			c->groups[g][n] = 0;
	}
	n = 0;
	while ((fscanf(fileptr, "%d", &g) != EOF)){
		c->groups[g][c->ngdocs[g]] = n;
		c->ngdocs[g] += 1;
		n += 1;
	}
	fclose(fileptr);

	return(c);
}

int max_corpus_length(M4_corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->ndocs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}


void write_M4_model(M4_model * model, char * root,M4_corpus * corpus)
{
    char filename[200];
    FILE* fileptr;
    int n, j, t;

    //beta
    sprintf(filename, "%s.beta", root);
    fileptr = fopen(filename, "w");
    for (n = 0; n < model->nterms; n++){
    	for (j = 0; j < model->ntopics; j++){
    		fprintf(fileptr, "%.10lf ", (model->logbeta[n][j]));
    	}
    	fprintf(fileptr, "\n");
    }
    fclose(fileptr);

    //pi
	sprintf(filename, "%s.pi", root);
	fileptr = fopen(filename, "w");
	for (t = 0; t < model->ngenres; t++){
		fprintf(fileptr, "%5.10lf ", log(model->pi[t]));
	}
	fclose(fileptr);

	//alpha
	sprintf(filename, "%s.alpha", root);
	fileptr = fopen(filename, "w");
	for (t = 0; t < model->ngenres; t++){
		for (j = 0; j < model->ntopics; j++){
			fprintf(fileptr, "%5.10lf ", log(model->alpha[t][j]));
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	sprintf(filename, "%s.other", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr,"num_topics %d \n",model->ntopics);
	fprintf(fileptr,"num_terms %d \n",model->nterms);
	fprintf(fileptr,"num_genres %d \n",model->ngenres);
	fclose(fileptr);

}

M4_model* load_model(char* model_root, int ngroups){

	char filename[100];
	FILE* fileptr;
	int j, n, num_topics, num_terms, num_genres, t;
	//float x;
	double y;

	M4_model* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_genres %d\n", &num_genres);
	fclose(fileptr);

	model = new_M4_model(num_topics, num_terms, ngroups, num_genres);

	sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    for (n = 0; n < num_terms; n++){
		for (j = 0; j < num_topics; j++){
			fscanf(fileptr, " %lf", &y);
			model->logbeta[n][j] = (y);
		}
	}
    fclose(fileptr);

	sprintf(filename, "%s.alpha", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
	for (t = 0; t < model->ngenres; t++){
		for (j = 0; j < model->ntopics; j++){
			fscanf(fileptr, " %lf", &y);
			model->alpha[t][j] = exp(y);
		}
	}
    fclose(fileptr);

	sprintf(filename, "%s.pi", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
	for (t = 0; t < model->ngenres; t++){
		fscanf(fileptr, " %lf", &y);
		model->pi[t] = exp(y);
	}
    fclose(fileptr);

    return(model);
}


void corpus_initialize_model(M4_model* model, M4_corpus* corpus, M4_ss* ss)
{

	int n, j, d, i, count, t, g, dd, argmaxlkh;
	double sum, temp, maxlkh;
	int* sdocs = malloc(sizeof(int)*corpus->ndocs);
	for (d = 0; d < corpus->ndocs; d++){
		sdocs[d] = -1;
	}
	//init topics
	for (j = 0; j < model->ntopics; j++){
		for (n = 0; n < model->nterms; n++){
			model->logbeta[n][j] = 1e-5;
		}
		sum = (double)model->nterms*(1e-5);
		for (i = 0; i < NUM_INIT; i++){
			//choose a doc from this tim and init
			count = 0;
			while (1){
				d = floor(gsl_rng_uniform(r) * corpus->ndocs);
				if(sdocs[d] != -1){
					count ++;
					continue;
				}
				else{
					sdocs[d] = j;
					break;
				}
			}

			for (n = 0; n < corpus->docs[d].length; n++){
				model->logbeta[corpus->docs[d].words[n]][j] += (double) corpus->docs[d].counts[n];
				sum += (double) corpus->docs[d].counts[n];
			}
		}
		for (n = 0; n < model->nterms; n++){
			model->logbeta[n][j] = log(model->logbeta[n][j]/sum);
		}
	}

	//init pi uniformly (and alpha)
	for (t = 0; t < model->ngenres; t++){
		model->pi[t] = 1.0/((double)model->ngenres);
		for (j = 0; j < model->ntopics; j++){
			model->alpha[t][j] = 0.1;
		}
	}

	//loop over docs in each group, hard-assign them to one group,
	// and init alpha by choosing one genre and averaging tpc assignments of the docs
	for (g = 0; g < model->ngroups; g++){

		t = floor(gsl_rng_uniform(r) * model->ngenres);

		for (dd = 0; dd < corpus->ngdocs[g]; dd++){
			d = corpus->groups[g][dd];
			maxlkh = -1e100;
			for (j = 0; j < model->ntopics; j++){
				temp = 0;
				for (n = 0; n < corpus->docs[d].length; n++){
					temp += (double)corpus->docs[d].counts[n]*(model->logbeta[corpus->docs[d].words[n]][j]);
				}
				if (temp > maxlkh){
					maxlkh = temp;
					argmaxlkh = j;
				}
			}
			model->alpha[t][argmaxlkh] += 1.0/((double)corpus->ngdocs[g]);
		}
	}

  	free(sdocs);


}



void random_initialize_model(M4_model * model, M4_corpus* corpus, M4_ss* ss){

	int n, j, t;
	double* beta = malloc(sizeof(double)*model->nterms);
	double* beta0 = malloc(sizeof(double)*model->nterms);
	double* pi = malloc(sizeof(double)*model->ngenres);
	double* pi0 = malloc(sizeof(double)*model->ngenres);

	for (n = 0; n < model->nterms; n++){
		beta[n] = 0.0;
		beta0[n] = 0.01;
	}
	for (t = 0; t < model->ngenres; t++){
		pi[t] = 0.0;
		pi0[t] = 1.0;
	}
	gsl_ran_dirichlet (r, model->ngenres, pi0, pi);
	for (t = 0; t < model->ngenres; t++){
		//model->pi[t] = pi[t];
		model->pi[t] = 1.0/((double)model->ngenres);
		for (j = 0; j < model->ntopics; j++){
			model->alpha[t][j] = gsl_rng_uniform(r);
		}
	}

	for (j = 0; j < model->ntopics; j++){

		gsl_ran_dirichlet (r, model->nterms, beta0, beta);
		for (n = 0; n < model->nterms; n++){
			model->logbeta[n][j] = log(beta[n]);
			beta[n] = 0.0;
		}
	}

  	free(beta);
  	free(pi);
  	free(beta0);
  	free(pi0);
}



/*
 * given log(a) and log(b), return log(a + b)
 *
 */

double log_sum(double log_a, double log_b)
{
  double v;

  if (log_a < log_b)
  {
      v = log_b+log(1 + exp(log_a-log_b));
  }
  else
  {
      v = log_a+log(1 + exp(log_b-log_a));
  }
  return(v);
}

