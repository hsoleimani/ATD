#include "main.h"



int main(int argc, char* argv[])
{

	char task[40];
	char dir[400];
	char corpus_file[400];
	char vdocsfile[400];
	char model_name[400];
	char titlefile[400];
	char thetafile[400];
	char doclenfile[400];
	int lmax;
	int BSSize;
    long int seed;

    AutoBreak = 0;
	MAXITER = 500;
	CONVERGED = 1e-4;
    NU = 0;

    seed = atol(argv[1]);
    printf("SEED = %ld\n", seed);
    seedMT(seed);

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);

    if (argc > 1)
    {

        if (strcmp(task, "detec")==0)
        {
			strcpy(dir,argv[4]);
			strcpy(model_name,argv[5]);
			strcpy(titlefile,argv[6]);
			lmax = (double) atof(argv[7]);
			strcpy(vdocsfile,argv[8]);
			AutoBreak = atoi(argv[9]);
			atd(corpus_file, dir, model_name, titlefile, lmax, vdocsfile);
            return(0);
        }
        if (strcmp(task, "btstp2")==0)
        {
			strcpy(dir,argv[4]);
			strcpy(model_name,argv[5]);
			strcpy(thetafile,argv[6]);
			lmax = atoi(argv[7]);
			BSSize = atoi(argv[8]);
			strcpy(doclenfile,argv[9]);
			btstp2(corpus_file, dir, model_name, thetafile, lmax, BSSize, doclenfile);
            return(0);
        }

    }
    return(0);
}


void btstp2(char* vdocsfile, char* dir, char* model_name, char* thetafile, int Ssize
		, int BSSize, char* doclenfile){

    FILE* fp;
    FILE* fplkh;
    char filename[400];
	int ntopics, nd, w;
	int nterms, j, d, n, nn, dd;
	double sum, lhood, lhood0, c;
    int ndmax;

	atd_var *var = NULL;
	atd_corpus* corpus;
	atd_corpus* valid_corpus;
    atd_model *model = NULL;
    atd_ss* ss;
    atd_valid_var *valid_var = NULL;

  	sprintf(filename, "%s.other", model_name);
  	printf("loading %s\n", filename);
  	fp = fopen(filename, "r");
  	fscanf(fp, "num_topics %d\n", &ntopics);
  	fscanf(fp, "num_terms %d\n", &nterms);
  	fclose(fp);

  	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

  	model = new_atd_model(ntopics, nterms, Ssize, model_name);

  	//**************************************************************
  	valid_corpus = read_data(vdocsfile, ntopics);
  	valid_var = new_atd_valid_var(valid_corpus, ntopics);
  	valid_nullmodel(valid_corpus, model, valid_var);


  	ndmax = 0;
	for (d = 0; d < valid_corpus->ndocs; d++){
		if (valid_corpus->docs[d].total > ndmax){
			ndmax = valid_corpus->docs[d].total;
		}
	}
  	// prepare BS corpus
  	corpus = malloc(sizeof(atd_corpus));
  	corpus->docs = malloc(sizeof(document) * Ssize);
  	corpus->ndocs = Ssize;
  	corpus->nterms = valid_corpus->nterms;
  	for (d = 0; d < Ssize; d++){
  		corpus->docs[d].tpcont = 0.0;
  		corpus->docs[d].length = 0;
  		corpus->docs[d].total = 0;
  		corpus->docs[d].words = malloc(sizeof(int)*ndmax);
  		corpus->docs[d].counts = malloc(sizeof(int)*ndmax);
  		for (n = 0; n < ndmax; n++){
  			corpus->docs[d].words[n] = 0;
  			corpus->docs[d].counts[n] = 0;
	    }
    }

  	//**************************************************************
    ss = new_atd_ss(model);
    var = new_atd_var(Ssize, ntopics, ndmax, model->n);

    // read theta0 of the clsuter
	printf("loading %s\n", thetafile);
	double x;
	double** theta0 = malloc(sizeof(double*)*Ssize);
	fp = fopen(thetafile, "r");
	for (d = 0; d < corpus->ndocs; d++){
		theta0[d] = malloc(sizeof(double)*ntopics);
		sum = 0.0;
		for (j = 0; j < ntopics; j++){
			fscanf(fp, "%lf ", &x);
			theta0[d][j] = x;
			sum += theta0[d][j];
		}
		for (j = 0; j < ntopics; j++){
			theta0[d][j] /= sum;
		}
	}
	fclose(fp);


	// read doclenfile
	int* doclen = malloc(sizeof(int)*Ssize);
	printf("loading %s\n", doclenfile);
	fp = fopen(doclenfile, "r");
	for (d = 0; d < Ssize; d++){
		fscanf(fp, "%d", &n);
		doclen[d] = n;
	}
	fclose(fp);



	//find similar docs
	double maxsim, sim;
	int** simdocs = malloc(sizeof(int*)*Ssize);
	int* numsimdocs = malloc(sizeof(int)*Ssize);
	double* maxmatch = malloc(sizeof(double)*Ssize);
	for (nd = 0; nd < Ssize; nd++){
		simdocs[nd] = malloc(sizeof(int)*valid_corpus->ndocs);
		numsimdocs[nd] = 0;
		maxsim = -1e10;
		for (d = 0; d < valid_corpus->ndocs; d++){
			sim = 0.0;
			for (j = 0; j < model->m; j++){
				if ((valid_var->v0[j][d] == 1) && (theta0[nd][j] != 0))
					sim += theta0[nd][j]*valid_var->theta0[j][d];
			}
			sim /= valid_var->normtheta0[d];
			if (sim > maxsim){
				simdocs[nd][0] = d;
				numsimdocs[nd] = 1;
				maxsim = sim;
			}
			else if(sim == maxsim){
				simdocs[nd][numsimdocs[nd]] = d;
				numsimdocs[nd] ++;
			}
		}
		maxmatch[nd] = maxsim;
	}

	////////// temp
	sprintf(filename, "%s_temp", thetafile);
	printf("saving %s\n", filename);
	fp = fopen(filename, "w");
	for (d = 0; d < Ssize; d++){
		fprintf(fp, "%lf: ", maxmatch[d]);
		for (j = 0; j < numsimdocs[d]; j++){
			fprintf(fp, "%d ", simdocs[d][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	///////

	int b, argmaxsim, mother_ld, ld, n0, alreadyin, nbs;
	double r, mother_nd;
	sprintf(filename, "%s/BSlkhratio.txt", dir);
	fplkh = fopen(filename, "w");
	fclose(fplkh);

	for (b = 0; b < BSSize; b++){

		//generate BS documents
		for (nd = 0; nd < Ssize; nd++){
			//randomly choose a similar doc
			if (numsimdocs[nd] == 1){
				argmaxsim = simdocs[nd][0];
			}
			else{
				r = myrand();
				for (d = 0; d < numsimdocs[nd]; d++){
					if (r <= (d+1.0)/((double)numsimdocs[nd])){
						argmaxsim = simdocs[nd][d];
						break;
					}
				}
			}

			//fill in pwords
			mother_ld = valid_corpus->docs[argmaxsim].length;
			mother_nd = (double) valid_corpus->docs[argmaxsim].total;
			valid_var->pwords[0] = (double) (valid_corpus->docs[argmaxsim].counts[0])/mother_nd;
			for (n = 1; n < mother_ld; n++){
				valid_var->pwords[n] = valid_var->pwords[n-1] +
						((double) valid_corpus->docs[argmaxsim].counts[n])/mother_nd;
			}
			//nd = mother_nd;
			ld = 0;
			for (n0 = 0; n0 < doclen[nd]; n0++){
				r = myrand();
				for (n = 0; n < mother_ld; n++){
					if (r <= valid_var->pwords[n]){
						nbs = n;
						break;
					}
				}
				alreadyin = 0;
				for (n = 0; n < ld; n++){
					if (valid_corpus->docs[argmaxsim].words[nbs] == corpus->docs[nd].words[n]){
						corpus->docs[nd].counts[n] += 1;
						alreadyin = 1;
						break;
					}
				}
				if (alreadyin == 0){
					corpus->docs[nd].words[ld] = valid_corpus->docs[argmaxsim].words[nbs];
					corpus->docs[nd].counts[ld] = 1;
					ld += 1;
				}
			}
			corpus->docs[nd].length = ld;
			corpus->docs[nd].total = doclen[nd];
		}

		nullmodel(corpus, model, ss, var);

		for (n = 0; n < model->n; n++){
			model->beta[ntopics][n] = 0.0;
			model->u[n] = 0;
			var->wrdchk[n] = 0;
		}
		model->mu = model->sumshared;
		model->zeta = 0.0;
		var->swrdlen = 0;
		var->slen = 0;
		sum = 0.0;
		var->sumnd = 0.0;
		for (d = 0; d < corpus->ndocs; d++){
			var->sdocs[var->slen] = d;
			var->docchk[d] = 1;
			var->slen += 1;

			var->sumnd += (double)corpus->docs[d].total;

			for (n = 0; n < corpus->docs[d].length; n++){
				c = (double) corpus->docs[d].counts[n];
				w = corpus->docs[d].words[n];

				//alternative model
				if (var->wrdchk[w] != 1){
					model->u[w] = 1;
					var->swrds[var->swrdlen] = w;
					var->swrdlen += 1;
					var->wrdchk[w] = 1;
					model->mu -= model->shared[w];
				}

				model->beta[ntopics][w] += c + NU;
				model->zeta += c + NU;
			}

			model->theta[ntopics][d] = 1.0;
			model->v[ntopics][d] = 1;
			for (j = 0; j < ntopics; j++){
				model->theta[j][d] = 0.0;//0.2/((double)ntopics);
				model->v[j][d] = 0;
			}
		}
		sum = model->zeta/(1-model->mu);
		for (n = 0; n < model->n; n++){
			if (model->u[n] == 1){
				model->beta[ntopics][n] = model->beta[ntopics][n]/sum;
			}else{
				model->beta[ntopics][n] = model->shared[n];
			}
		}

		lhood = TrainAltModel(corpus, model, ss, var);

		lhood0 = 0.0;
		lhood = 0.0;
		for (dd = 0; dd < var->slen; dd++){
			d = var->sdocs[dd];
			lhood0 += var->lkh0[d];///((double)corpus->docs[d].total);
			lhood += var->lkh[d];///((double)corpus->docs[d].total);
		}
		sum = 0;
		for (nn = 0; nn < var->swrdlen; nn++){
			sum += model->u[var->swrds[nn]];
		}

		fplkh = fopen(filename, "a");
		fprintf(fplkh,"%lf\t%d\t%d\n",lhood-lhood0,var->swrdlen,(int)sum);
		fclose(fplkh);

		if ((b%100) == 0)
			printf("Btstp step %d \n", b);

	}

}



void atd(char* dataset, char* dir, char* model_name, char* titlefile, int lmax, char* vdocsfile){

    FILE* fp;
    FILE* fplkh;
    char filename[400];
    char lkhinc[400];
    char tpcont[400];
    FILE* fp1;
    FILE* fp3;
	int nmax, ntopics;
	int nterms, j, d, n, l, nn, dd, w;
	double sum, lhood, lhood0, pval;
	double linc;//, conv,t1, t2;

	atd_var *var = NULL;
	atd_corpus* corpus;
	atd_corpus* valid_corpus;
    atd_model *model = NULL;
    atd_ss* ss;
    atd_valid_var *valid_var = NULL;

  	sprintf(filename, "%s.other", model_name);
  	printf("loading %s\n", filename);
  	fp = fopen(filename, "r");
  	fscanf(fp, "num_topics %d\n", &ntopics);
  	fscanf(fp, "num_terms %d\n", &nterms);
  	fclose(fp);

  	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

  	corpus = read_data(dataset, ntopics);

  	if (lmax == 0)	lmax = corpus->ndocs;

  	printf("loading %s\n", titlefile);
  	fp = fopen(titlefile, "r");
  	for (d = 0; d < corpus->ndocs; d++){
  		fscanf(fp,"%s\n",corpus->docs[d].title);
  	}
  	fclose(fp);

  	nmax = max_corpus_length(corpus);

  	model = new_atd_model(ntopics, nterms, corpus->ndocs, model_name);
    ss = new_atd_ss(model);
    var = new_atd_var(corpus->ndocs, ntopics, nmax, model->n);

  	valid_corpus = read_data(vdocsfile, ntopics);
  	valid_var = new_atd_valid_var(valid_corpus, ntopics);
  	valid_nullmodel(valid_corpus, model, valid_var);

    //learn theta0 under null model
    nullmodel(corpus, model, ss, var);

    //write normalized lkh0
    sprintf(filename, "%s/test.normlkh0", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		fprintf(fp,"%5.5lf\n",var->lkh0[d]/((double)corpus->docs[d].total));
	}
	fclose(fp);

    //write theta0
	sprintf(filename, "%s/test.theta0", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			if (j >0)
				fprintf(fp,",%5.6lf",model->theta0[j][d]);
			else
				fprintf(fp,"%5.6lf",model->theta0[j][d]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);

	printf("Done with learning on the null model, first doc = %d\n", var->firstdoc);


	var->slen = 0;
    var->sdocs[var->slen] = var->firstdoc;
    var->docchk[var->firstdoc] = 1;
    var->slen = 1;

    //******
	sprintf(tpcont, "%s/last.tpcont", dir);
	fp1 = fopen(tpcont, "w+");
	fprintf(fp1,"%lf\t%lf\n",corpus->docs[var->firstdoc].tpcont, 1.0);
	fclose(fp1);
	//******

	// train beta, theta, u on first doc
    //init
    d = var->firstdoc;
    var->sumnd = (double)corpus->docs[d].total;
    model->mu = 0.0;//model->sumshared;
    for (n = 0; n < model->n; n++){
    	model->u[n] = 0;
    	model->mu += model->shared[n];
    }
    var->swrdlen = 0;
    model->zeta = 0.0;
    for (n = 0; n < corpus->docs[d].length; n++){
    	if (var->wrdchk[corpus->docs[d].words[n]] == 0){
    		var->swrds[var->swrdlen] = corpus->docs[d].words[n];
    		var->swrdlen += 1;
    		var->wrdchk[corpus->docs[d].words[n]] = 1;

    		model->u[corpus->docs[d].words[n]] = 1;
    		model->betahat[corpus->docs[d].words[n]] = 0.0;
    		model->mu -= model->shared[n];
    	}
    	model->betahat[corpus->docs[d].words[n]] += (double)corpus->docs[d].counts[n] + NU;
    	model->zeta += (double)corpus->docs[d].counts[n] + NU;
    }
    sum = model->zeta/(1.0-model->mu);
	for (nn = 0; nn < var->swrdlen; nn++){
		n = var->swrds[nn];
    	if (model->u[n] == 1)
    		model->beta[ntopics][n] = model->betahat[n]/sum;
    	else
    		model->beta[ntopics][n] = model->shared[n];
    }

	model->theta[ntopics][d] = 1;
	model->v[ntopics][d] = 1;
	for (j = 0; j < ntopics; j++){
		model->theta[j][d] = 0.0;//0.2/((double)ntopics);
		model->v[j][d] = 0;
	}

    //train
   	lhood0 = TrainAltModel(corpus, model, ss, var);

    sprintf(filename, "%s/lkhratio.txt", dir);

	lhood0 = 0.0;
	lhood = 0.0;
	for (dd = 0; dd < var->slen; dd++){
		d = var->sdocs[dd];
		lhood0 += var->lkh0[d];
		lhood += var->lkh[d];
	}
	sum = 0;
	for (nn = 0; nn < var->swrdlen; nn++){
		sum += model->u[var->swrds[nn]];
	}
	fplkh = fopen(filename, "w");
	fprintf(fplkh,"%d\t%lf\t%d\t%d\n",1,lhood-lhood0,var->swrdlen,(int)sum);
	fclose(fplkh);

	write_atd_model(model, dir, corpus, var);


	lhood = 0.0;
    // main loop
	int stpcnt = 0;
	for (l = 1; l < lmax; l++){ // l is the total size of cluster (max = #docs)

		lhood0 = LearnUnderAltModel(corpus, model, ss, var);

		//************** Run to BS to determine significance of this sample

		linc = (var->lkh[var->firstdoc]-var->lkh0[var->firstdoc])/fabs(var->lkh0[var->firstdoc]);

		if ((AutoBreak == 1) && (corpus->docs[var->firstdoc].tpcont <= 0.2)){
			pval = SingleDocBtstp(&(corpus->docs[var->firstdoc]), valid_corpus, valid_var, model, var->firstdoc);
		}
		else
			pval = 1.0;

		//************** End: Run to BS to determine significance of this sample

        //check sopping rule
		if (l > 4){
			if ((corpus->docs[var->firstdoc].tpcont <= 0.2) && (pval <= 0.99)){
				stpcnt ++;
			}else{
				stpcnt = 0;
			}
		}

		//******
    	sprintf(tpcont, "%s/last.tpcont", dir);
    	fp1 = fopen(tpcont, "a");
    	fprintf(fp1,"%lf\t%lf\n",corpus->docs[var->firstdoc].tpcont, pval);
    	fclose(fp1);

    	sprintf(lkhinc, "%s/last.lkhinc", dir);
    	fp3 = fopen(lkhinc, "a");
    	fprintf(fp3,"%lf\n",linc);
    	fclose(fp3);
        //******

    	if (stpcnt == 2){
    		printf("******************** Break ******************* \n");
    		if (AutoBreak == 1)
    			break;
    	}

    	//add the doc selected in this loop to S
        var->sdocs[var->slen] = var->firstdoc;
        var->docchk[var->firstdoc] = 1;
        var->slen += 1;

        // train beta, theta, u on first doc

        for (n = 0; n < model->n; n++){
        	model->u[n] = 0;
        	model->beta[ntopics][n] = 0.0;
        }
        model->mu = model->sumshared;
        model->zeta = 0.0;
		d = var->firstdoc;

		for (n = 0; n < corpus->docs[d].length; n++){ //add new words from the new doc to the list
			w = corpus->docs[d].words[n];
			if (var->wrdchk[w] == 0){
				var->swrds[var->swrdlen] = w;
				var->swrdlen += 1;
				var->wrdchk[w] = 1;
			}
		}
		var->sumnd = 0.0;
		for (dd = 0; dd < var->slen; dd++){ // init beta
			d = var->sdocs[dd];
			var->sumnd += (double)corpus->docs[d].total;
			for (n = 0; n < corpus->docs[d].length; n++){
				w = corpus->docs[d].words[n];
				if (model->u[w] == 0) //needs u to be initially zero (only subtract 1st occurrence of w)
					model->mu -= model->shared[w];
				model->u[w] = 1;
				model->beta[ntopics][w] += (double) corpus->docs[d].counts[n] + NU;
				model->zeta += (double) corpus->docs[d].counts[n] + NU;
			}
			model->theta[ntopics][d] = 1;
			model->v[ntopics][d] = 1;
			for (j = 0; j < ntopics; j++){
				model->theta[j][d] = 0.0;
				model->v[j][d] = 0;
			}
		}
		sum = model->zeta/(1-model->mu);
		for (n = 0; n < model->n; n++){
			if (model->u[n] == 1){
				model->beta[ntopics][n] = model->beta[ntopics][n]/sum;
			}else{
				model->beta[ntopics][n] = model->shared[n];
			}
		}

		// retrain beta, theta, and u
		lhood = TrainAltModel(corpus, model, ss, var);

    	lhood0 = 0.0;
    	lhood = 0.0;
		for (dd = 0; dd < var->slen; dd++){
			d = var->sdocs[dd];
			lhood0 += var->lkh0[d];///((double)corpus->docs[d].total);
			lhood += var->lkh[d];///((double)corpus->docs[d].total);
		}
		sum = 0;
		for (nn = 0; nn < var->swrdlen; nn++){
			sum += model->u[var->swrds[nn]];
		}
		fplkh = fopen(filename, "a");
		fprintf(fplkh,"%d\t%lf\t%d\t%d\n",l+1,lhood-lhood0,var->swrdlen,(int)sum);
    	fclose(fplkh);

    	if (stpcnt == 0)
    		write_atd_model(model, dir, corpus, var);

    	if ((l%1) == 0){
    		printf("%d, linc = %lf, tpcont = %lf, pval = %lf \n",l, linc,
    				corpus->docs[var->firstdoc].tpcont,pval);
    	}

    }

}


atd_valid_var * new_atd_valid_var(atd_corpus* valid_corpus, int ntopics)
{
  	int j, ndmax, ldmax, d, n;
	atd_valid_var *valid_var = malloc(sizeof(atd_valid_var));

  	ndmax = 0;
  	ldmax = 0;
  	valid_var->B = 1000;
  	valid_var->phi = malloc(sizeof(double)*(ntopics+1));
  	valid_var->theta = malloc(sizeof(double)*(ntopics+1));
  	valid_var->temp_theta = malloc(sizeof(double)*(ntopics+1));
  	valid_var->sstheta = malloc(sizeof(double)*(ntopics+1));
  	valid_var->v = malloc(sizeof(int)*(ntopics+1));
  	for (j = 0; j < (ntopics+1); j++){
  		valid_var->phi[j] = 0.0;
  		valid_var->theta[j] = 0.0;
  		valid_var->temp_theta[j] = 0.0;
  		valid_var->sstheta[j] = 0.0;
  		valid_var->v[j] = 0;
  	}
  	valid_var->theta0 = malloc(sizeof(double*)*ntopics);
  	valid_var->v0 = malloc(sizeof(int*)*ntopics);
  	for (j = 0; j < ntopics; j++){
  		valid_var->theta0[j] = malloc(sizeof(double)*valid_corpus->ndocs);
  		valid_var->v0[j] = malloc(sizeof(int)*valid_corpus->ndocs);
  		if (j == 0){
  			valid_var->normtheta0 = malloc(sizeof(double)*valid_corpus->ndocs);
  			valid_var->simdocs = malloc(sizeof(int)*valid_corpus->ndocs);
  			ndmax = 0;
  			ldmax = 0;
  		}
  		for (d = 0; d < valid_corpus->ndocs; d++){
  			valid_var->theta0[j][d] = 0.0;
  			valid_var->v0[j][d] = 0;
  			if (j == 0){
  				valid_var->normtheta0[d] = 0.0;
  				valid_var->simdocs[d] = 0;
  				if (valid_corpus->docs[d].total > ndmax){
  					ndmax = valid_corpus->docs[d].total;
  				}
  				if (valid_corpus->docs[d].total > ldmax){
  					ldmax = valid_corpus->docs[d].total;
  				}
  			}
  		}
  	}
  	/////
    valid_var->doc = 0;
    valid_var->doc = (document*) realloc(valid_var->doc, sizeof(document)*(1));
    valid_var->doc[0].tpcont = 0.0;
    valid_var->doc[0].length = 0;
    valid_var->doc[0].total = 0;
    valid_var->doc[0].words = malloc(sizeof(int)*ndmax);
    valid_var->doc[0].counts = malloc(sizeof(int)*ndmax);
  	for (n = 0; n < ndmax; n++){
  		valid_var->doc[0].words[n] = 0;
  		valid_var->doc[0].counts[n] = 0;
  	}
  	/////

  	valid_var->pwords = malloc(sizeof(double)*ldmax);
  	for (n = 0; n < ldmax; n++){
  		valid_var->pwords[n] = 0.0;
  	}

  	return(valid_var);
}



double SingleDocBtstp(document* doc, atd_corpus* valid_corpus, atd_valid_var* valid_var,
		atd_model* model, int d0){

	double pval, sim, maxsim, r;
	int d, j, n, b, ld, n0, numsimdocs, argmaxsim;
	int mother_ld, alreadyin, nbs, ntopics, w;
	double mother_nd, c, temp;
	double BStpcnt, phimax;
	int argphimax;

	ntopics = model->m + 1;

	numsimdocs = 0;
	maxsim = -1e10;
	for (d = 0; d < valid_corpus->ndocs; d++){
		sim = 0.0;
		for (j = 0; j < model->m; j++){
			if ((valid_var->v0[j][d] == 1) && (model->v0[j][d0] == 1))
				sim += model->theta0[j][d0]*valid_var->theta0[j][d];
		}
		sim /= valid_var->normtheta0[d];

		if (sim > maxsim){
			valid_var->simdocs[0] = d;
			numsimdocs = 1;
			maxsim = sim;
		}
		else if(sim == maxsim){
			valid_var->simdocs[numsimdocs] = d;
			numsimdocs ++;
		}
	}

	pval = 0.0;
	//main BS loop
	for (b = 0; b < valid_var->B; b++){

		// *** generate BS doc
		//randomly choose a similar doc
		if (numsimdocs == 1){
			argmaxsim = valid_var->simdocs[0];
		}
		else{
			r = myrand();
			for (d = 0; d < numsimdocs; d++){
				if (r <= (d+1.0)/((double)numsimdocs)){
					argmaxsim = valid_var->simdocs[d];
					break;
				}
			}
		}
		//fill in pwords
		mother_ld = valid_corpus->docs[argmaxsim].length;
		mother_nd = (double) valid_corpus->docs[argmaxsim].total;
		valid_var->pwords[0] = (double) (valid_corpus->docs[argmaxsim].counts[0])/mother_nd;
		for (n = 1; n < mother_ld; n++){
			valid_var->pwords[n] = valid_var->pwords[n-1] +
					((double) valid_corpus->docs[argmaxsim].counts[n])/mother_nd;
		}
		ld = 0;
		for (n0 = 0; n0 < doc->total; n0++){
			r = myrand();
			for (n = 0; n < mother_ld; n++){
				if (r <= valid_var->pwords[n]){
					nbs = n;
					break;
				}
			}
			alreadyin = 0;
			for (n = 0; n < ld; n++){
				if (valid_corpus->docs[argmaxsim].words[nbs] == valid_var->doc[0].words[n]){
					valid_var->doc[0].counts[n] += 1;
					alreadyin = 1;
					break;
				}
			}
			if (alreadyin == 0){
				valid_var->doc[0].words[ld] = valid_corpus->docs[argmaxsim].words[nbs];
				valid_var->doc[0].counts[ld] = 1;
				ld += 1;
			}
		}

		valid_var->doc[0].length = ld;
		valid_var->doc[0].total = doc->total;

		// **** learn theta
		for (j = 0; j < ntopics; j++){
			valid_var->theta[j] = 0.0;
			valid_var->v[j] = 0;
		}
		valid_var->theta[model->m] = 1.0;
		valid_var->v[model->m] = 1;

		update_alt_theta(valid_var->theta, valid_var->temp_theta, valid_var->sstheta,
				valid_var->v, valid_var->phi, model, &(valid_var->doc[0]));


		BStpcnt = 0.0;
		for (n = 0; n < ld; n++){
			c = (double) valid_var->doc[0].counts[n];
			w = valid_var->doc[0].words[n];
			phimax = -1e10;
			argphimax = 0;
			for (j = 0; j < ntopics; j++){
				if (valid_var->v[j] == 1){
					temp = valid_var->theta[j]*model->beta[j][w];
					if (temp > phimax){
						phimax = temp;
						argphimax = j;
					}
				}
			}
			if (argphimax == model->m)	BStpcnt += c;
		}
		BStpcnt /= ((double)valid_var->doc[0].total);

		if (BStpcnt < doc->tpcont)
			pval += 1.0;
	}

	pval /= ((double)valid_var->B);
	return(pval);
}


double LearnUnderAltModel(atd_corpus* corpus, atd_model* model, atd_ss* ss, atd_var* var){

	int j, d, ntopics, w, n;
	double c, phisum;
	double varlkh, temp;
	double maxinc, lkhchange;

	double* theta0 = malloc(sizeof(double)*(model->m+1));
	double* theta1 = malloc(sizeof(double)*(model->m+1));
	int* v0 = malloc(sizeof(int)*(model->m+1));
	double* phi = malloc(sizeof(double)*(model->m+1));
	double* sstheta = malloc(sizeof(double)*(model->m+1));
	double phimax;
	int argphimax;
	ntopics = model->m + 1;
	maxinc = -1e40;

	//estep
	for (d = 0; d < corpus->ndocs; d++){
		//if already in S, skip
		if (var->docchk[d] == 1){
			continue;
		}

		//*********************** with v update
		//init with only new topic on

		for (j = 0; j < ntopics; j++){
			theta0[j] = 0.0;
			v0[j] = 0;
		}
		theta0[model->m] = 1.0;
		v0[model->m] = 1;

		update_alt_theta(theta0, theta1, sstheta, v0, phi, model, &(corpus->docs[d]));

		for (j = 0; j < ntopics; j++){
			if (v0[j] == 0)
				model->theta[j][d] = 0.0;
			else
				model->theta[j][d] = theta0[j];
			model->v[j][d] = v0[j];
		}


		//compute lkh with updated v and theta
		//compute tpc cont
		varlkh = 0.0;
		corpus->docs[d].tpcont = 0.0;
		for (n = 0; n < corpus->docs[d].length; n++){
			c = (double) corpus->docs[d].counts[n];
			w = corpus->docs[d].words[n];
			phimax = -1e10;
			argphimax = 0;
			phisum = 0.0;
			for (j = 0; j < ntopics; j++){
				if (model->v[j][d] == 1){
					temp = model->theta[j][d]*model->beta[j][w];
					phisum += temp;
					if (temp > phimax){
						phimax = temp;
						argphimax = j;
					}
				}
			}
			if (argphimax == model->m)	corpus->docs[d].tpcont += c;
			varlkh += c*log(phisum);
		}
		corpus->docs[d].tpcont = corpus->docs[d].tpcont/((double)corpus->docs[d].total);
		var->lkh[d] = varlkh;///((double)corpus->docs[d].total);

		//compute change in lkh wrt null model
		lkhchange = (varlkh - var->lkh0[d])/fabs(var->lkh0[d]);
		if (lkhchange > maxinc){
			maxinc = lkhchange;
			var->firstdoc = d;
		}
	}

	free(phi);
	free(sstheta);
	free(theta0);
	free(theta1);
	free(v0);

	return(var->lkh[var->firstdoc]);
}



double TrainAltModel(atd_corpus* corpus, atd_model* model, atd_ss* ss, atd_var* var){

	int j, d, dd, ntopics, w, n, current, jj;
	double c, temp, cphi, phisum, sumtheta;
	double varlkh, deltaq;
	int iter, change, variter, variter2, sumon;
	double conv, lkh, prev_lkh, varlkh2, prev_lkh2, conv2;
	double phimax, mu, dl, dcost, new_mu, t1;
	double p1, ent_curr, ent_next, t2;
	double * theta0 = malloc(sizeof(double)*(model->m+1));
	double * theta1 = malloc(sizeof(double)*(model->m+1));
	int * v0 = malloc(sizeof(int)*(model->m+1));
	double * phi = malloc(sizeof(double)*(model->m+1));
	double * sstheta = malloc(sizeof(double)*(model->m+1));

	ntopics = model->m + 1;
	iter = 0;
	variter = 0;
	prev_lkh = -1e50;
	do{
		//*** zero init ss
		ss->sum_beta = (1e-5)*model->n;
		for (n = 0; n < model->n; n++){
			ss->beta[n] = 1e-5;//EPS;
		}

		//*** estep + update theta
		varlkh = 0.0;
		for (dd = 0; dd < var->slen; dd++){
			d = var->sdocs[dd];
			variter2 = 0;
			prev_lkh2 = 0.0;

			do{
				varlkh2 = 0.0;
				sumtheta = 0.0;
				for (j = 0; j < ntopics; j++){
					ss->theta[j][d] = 0.0;
				}

				for (n = 0; n < corpus->docs[d].length; n++){
					c = (double) corpus->docs[d].counts[n];
					w = corpus->docs[d].words[n];

					phisum = 0.0;
					for (j = 0; j < ntopics; j++){
						if (model->v[j][d] == 0){
							var->phi[n][j] = 0;
							continue;
						}
						var->phi[n][j] = model->theta[j][d]*model->beta[j][w];
						phisum += var->phi[n][j];
					}

					for (j = 0; j < ntopics; j++){
						if (model->v[j][d] == 0){
							continue;
						}
						temp = var->phi[n][j];
						var->phi[n][j] /= phisum;
						cphi = c*var->phi[n][j];

						if (var->phi[n][j] > 0){
							varlkh2 += cphi*(log(temp) - log(var->phi[n][j])); //8
							//update ss
							ss->theta[j][d] += cphi;
							sumtheta += cphi;
						}
					}
				}
				//update theta
				for (j = 0; j < ntopics; j++){
					if (model->v[j][d] == 1)
						model->theta[j][d] = ss->theta[j][d]/sumtheta;
					else
						model->theta[j][d] = 0.0;
				}

				conv2 = (prev_lkh2 - varlkh2)/prev_lkh2;
				prev_lkh2 = varlkh2;
				variter2 ++;

			}while((variter2 < MAXITER) && ((conv2 > CONVERGED) || (conv2 < 0)));

			varlkh += varlkh2;

			//update ss
			//corpus->docs[d].tpcont = 0.0;
			for (n = 0; n < corpus->docs[d].length; n++){
				c = (double) corpus->docs[d].counts[n];
				w = corpus->docs[d].words[n];

				phimax = -1e10;
				for (j = 0; j < ntopics; j++){

					if (var->phi[n][j] > phimax){
						phimax = var->phi[n][j];
					}
					if (var->phi[n][j] == 0){
						continue;
					}
					cphi = c*var->phi[n][j];
					if (j == model->m){
						ss->beta[w] += cphi;
						ss->sum_beta += cphi;
					}
				}
			}
		}
		//*** mstep
		//update beta
		model->mu = model->sumshared;
		model->zeta = 0.0;
		for(n = 0; n < model->n; n++){
			if (model->u[n] == 1){
				model->mu -= model->shared[n];
				model->zeta += ss->beta[n];
			}
		}
		mu = model->zeta/(1.0-model->mu);
		sumon = 0;
		for (n = 0; n < model->n; n++){
			if (model->u[n] == 0)
				model->beta[model->m][n] = model->shared[n];
			else{
				model->beta[model->m][n] = ss->beta[n]/mu;
				sumon += 1;
			}
		}

		// ******** ********* update u
		iter = 0;
		do{
			//break;
			change = 0;
			for (n = 0; n < model->n; n++){
				current = model->u[n];

				if ((current == 1) && (sumon <= 2)) continue; //this is the only ''on'' switch; skip

				model->u[n] = 1-current; //temp. flip this switch

				if (current == 0){
					new_mu = (model->zeta + ss->beta[n])/(1-model->mu + model->shared[n]);
				}else{
					new_mu = (model->zeta - ss->beta[n])/(1-model->mu - model->shared[n]);
				}
				t1 = 0.5*log(var->sumnd/2.0/PI);
				p1 = ((double)sumon)/((double)model->n);
				ent_curr = -p1*log(p1)-(1-p1)*log(1-p1);
				p1 = ((double)sumon + 1.0-2.0*current)/((double)model->n);
				ent_next = -p1*log(p1)-(1-p1)*log(1-p1);
				t2 = model->n*log(2.0)*(ent_next-ent_curr);
				if (current == 0)
					dcost = t1 + t2;
				else
					dcost = -t1 + t2;

				//delta lkh
				if (current == 0){
					if (ss->beta[n] == 0)
						dl = - model->zeta*log(new_mu/mu);
					else
						dl = ss->beta[n]*log(ss->beta[n]/(new_mu*model->shared[n]))
									- model->zeta*log(new_mu/mu);
				}else{
					if (ss->beta[n] == 0)
						dl =- model->zeta*log(new_mu/mu);
					else
						dl = -ss->beta[n]*log(ss->beta[n]/(mu*model->shared[n]))
							- model->zeta*log(new_mu/mu);
				}

				//delta bic
				deltaq = dcost - dl;
				if (deltaq < 0){ //accept

					if (current == 0){
						model->zeta += ss->beta[n];
						model->mu -= model->shared[n];
						sumon ++;
					}else{
						model->zeta -= ss->beta[n];
						model->mu += model->shared[n];
						sumon --;
					}
					mu = model->zeta/(1-model->mu);
					change ++;
				}else{ //reject
					mu = model->zeta/(1-model->mu);
					model->u[n] = current;
				}
			}

			iter ++;
		}while((iter < 10) && (change > 0));

		mu = model->zeta/(1.0-model->mu);
		sumon = 0;
		for (n = 0; n < model->n; n++){
			if (model->u[n] == 0)
				model->beta[model->m][n] = model->shared[n];
			else{
				model->beta[model->m][n] = ss->beta[n]/mu;
				sumon += 1;
			}
		}


		//  ***** update v
		for (dd = 0; dd < var->slen; dd++){
			d = var->sdocs[dd];


			for (jj = 0; jj < ntopics; jj++){
				theta0[jj] = model->theta[jj][d];
				 v0[jj] = model->v[jj][d];
			}

			update_alt_theta(theta0, theta1, sstheta, v0, phi, model, &(corpus->docs[d]));

			for (jj = 0; jj < ntopics; jj++){
				if (v0[jj] == 0)
					model->theta[jj][d] = 0.0;
				else
					model->theta[jj][d] = theta0[jj];
				model->v[jj][d] = v0[jj];
			}

		}

		lkh = compute_likelihood(corpus, var, model);
		//lkh = varlkh;
		conv = (prev_lkh - lkh)/prev_lkh;
		if ((conv < 0) && (change > 0))
			printf("****negative conv, %lf, %lf, %d %d, doc=%d, change=%d, %d\n",
					prev_lkh, lkh, variter, sumon, d, change, model->u[0]);
		prev_lkh = lkh;
		variter ++;

	}while((variter < MAXITER) && ((conv > CONVERGED) || (conv < 0)));

	free(theta0);
	free(theta1);
	free(v0);
	free(sstheta);
	free(phi);

	return(lkh);

}


//update alt topic proportions
void update_alt_theta(double* theta0, double* theta1, double* sstheta,
		int* v0, double* phi, atd_model* model, document* doc)
{

	int iter, sumon, change, j, n, w, current, jj, ntopics;
	int variter;
	double t1, t2, dcost, temp, sumtheta, c, cphi, conv, dl, deltaq;
	double varlkh, prev_lkh, temp_varlkh, phisum, temp_phisum;

	ntopics  = model->m + 1;
	iter = 0;
	sumon = 0;
	for (j = 0; j < ntopics; j++){
		sumon += v0[j];
	}
	do{
		change = 0;

		for (j = 0; j < model->m; j++){ //for now, assume topic M+1 is always on
			current = v0[j];

			if ((sumon == 1) && (current == 1)) continue;
			//temp change v
			v0[j] = 1-current;
			//dcost
			temp = sumon + 1.0 - 2*current;
			t1 = -lgamma(temp+1.0)-lgamma(ntopics-temp+1.0);
			t1 -= -lgamma(sumon+1.0)-lgamma(ntopics-sumon+1.0);
			t2 = 0.5*log((double)doc->total/2.0/PI);
			if (current == 1)	t2 = -t2;
			dcost = t1 + t2;

			//dl
			//temp-learn new theta
			sumtheta = (double) sumon + 1.0 - 2*current;
			for (jj = 0; jj < ntopics; jj++){
				if (v0[jj] == 1)
					theta1[jj] = 1.0/sumtheta;
				else
					theta1[jj] = 0.0;
			}

			variter = 0;
			prev_lkh = -1e50;
			sumtheta = 0.0;
			for (jj = 0; jj < ntopics; jj++){
				sstheta[jj] = 0.0;
			}
			do{ // (can remove this loop) loop to update theta
				varlkh = 0.0;
				temp_varlkh = 0.0;

				for (n = 0; n < doc->length; n++){
					c = (double) doc->counts[n];
					w = doc->words[n];
					phisum = 0.0;
					temp_phisum = 0.0;
					for (jj = 0; jj < ntopics; jj++){
						if (jj != j){
							if (v0[jj] == 1)
								phisum += theta0[jj]*model->beta[jj][w];
						}else{
							if (current == 1)
								phisum += theta0[jj]*model->beta[jj][w];
						}
						if (v0[jj] == 1)
							phi[jj] = theta1[jj]*model->beta[jj][w];
						else
							phi[jj] = 0;
						temp_phisum += phi[jj];
					}
					varlkh += c*log(phisum);

					for (jj = 0; jj < ntopics; jj++){

						temp = phi[jj];
						phi[jj] = phi[jj]/temp_phisum;
						cphi = c*phi[jj];
						if (phi[jj] > 0){
							temp_varlkh += cphi*(log(temp) - log(phi[jj]));
							sstheta[jj] += cphi;
							sumtheta += cphi;
						}
					}
				}
				//update theta
				for (jj = 0; jj < ntopics; jj++){
					theta1[jj] = sstheta[jj]/sumtheta;
					sstheta[jj] = 0.0;
				}
				sumtheta = 0.0;

				conv = (prev_lkh - temp_varlkh)/prev_lkh;
				prev_lkh = temp_varlkh;
				variter ++;

			}while((variter < MAXITER) && ((conv > CONVERGED) || (conv < 0)));
			dl = temp_varlkh - varlkh;

			deltaq = dcost - dl;
			if (deltaq < 0){ //accept
				for (jj = 0; jj < ntopics; jj++){
					theta0[jj] = theta1[jj];
				}
				sumon += 1 - 2*current;
				change ++;
			}else{ // reject
				v0[j] = current;
			}

		}
		iter ++;
	}while((iter < 10) && (change > 0));

}


double compute_likelihood(atd_corpus* corpus, atd_var* var, atd_model* model){

	double lkh, c, phisum, doclkh;
	//double logt, log1mt;
	int dd, d, j, n, w, ntopics;
	double p1, sumon, ent_curr;

	ntopics = model->m + 1;

	lkh = 0.0;
	for (dd = 0; dd < var->slen; dd++){
		doclkh = 0.0;

		d = var->sdocs[dd];

		for (n = 0; n < corpus->docs[d].length; n++){
			c = (double) corpus->docs[d].counts[n];
			w = corpus->docs[d].words[n];

			phisum = 0.0;
			for (j = 0; j < ntopics; j++){
				if (model->v[j][d] == 0)
					continue;
				phisum += model->theta[j][d]*model->beta[j][w];
			}
			doclkh += c*log(phisum);
		}
		var->lkh[d] = doclkh;

		//cost
		sumon = 0;
		for (j = 0; j < ntopics; j++){
			sumon += model->v[j][d];
		}
		doclkh += lgamma(ntopics+1.0) - lgamma(sumon+1.0)-lgamma(ntopics-sumon+1.0);
		doclkh += 0.5*sumon*log((double)corpus->docs[d].total/2.0/PI);

		lkh += doclkh;
	}

	lkh = -lkh;
	sumon = 0;
	for (n = 0; n < model->n; n++){
		sumon += model->u[n];
	}
	p1 = sumon/((double)model->n);
	ent_curr = -p1*log(p1)-(1-p1)*log(1-p1);
	lkh += sumon*0.5*log(var->sumnd/2.0/PI)+((double)model->n)*log(2.0)*ent_curr;

	return(lkh);
}


void valid_nullmodel(atd_corpus* corpus, atd_model* model, atd_valid_var* var){

	int d, j, n, w, argmaxlkh;
	double temp, c, maxlkh;
	//double ent_curr, ent_next, t2;
	double* theta0 = malloc(sizeof(double)*model->m);
	double* theta1 = malloc(sizeof(double)*model->m);
	int* v0 = malloc(sizeof(int)*model->m);
	double * sstheta = malloc(sizeof(double)*model->m);
	double * phi = malloc(sizeof(double)*model->m);

	for (d = 0; d < corpus->ndocs; d++){

		//compute lkh under all topics (only one present at a time)
		maxlkh = -1e100;
		for (j = 0; j < model->m; j++){
			theta0[j] = 0.0;
			v0[j] = 0;
			temp = 0.0;
			for (n = 0; n < corpus->docs[d].length; n++){
				c = (double) corpus->docs[d].counts[n];
				w = corpus->docs[d].words[n];
				temp += c*log(model->beta0[j][w]);
			}
			if (temp > maxlkh){
				maxlkh = temp;
				argmaxlkh = j;
			}
		}
		theta0[argmaxlkh] = 1.0;
		v0[argmaxlkh] = 1;

		//optimize theta
		update_null_theta(theta0, theta1, sstheta, v0, phi, model, &(corpus->docs[d]));

		var->normtheta0[d] = 0.0;
		for (j = 0; j < model->m; j++){
			var->theta0[j][d] = theta0[j];
			var->v0[j][d] = v0[j];
			if (var->v0[j][d] == 0){
				var->theta0[j][d] = 0.0;
				continue;
			}
			var->normtheta0[d] += var->theta0[j][d]*var->theta0[j][d];
		}
		var->normtheta0[d] = sqrt(var->normtheta0[d]);

	}

	free(theta0);
	free(theta1);
	free(v0);
	free(sstheta);
	free(phi);
}


//update null topic proportions
void update_null_theta(double* theta0, double* theta1, double* sstheta,
		int* v0, double* phi, atd_model* model, document* doc)
{

	int iter, sumon, change, j, n, w, current, jj;
	int variter;
	double t1, t2, dcost, temp, sumtheta, c, cphi, conv, dl, deltaq;
	double varlkh, prev_lkh, temp_varlkh, phisum, temp_phisum;

	iter = 0;
	sumon = 1;
	do{
		change = 0;
		for (j = 0; j < model->m; j++){
			current = v0[j];
			if ((sumon == 1) && (current == 1)) continue;

			//temp change v
			v0[j] = 1-current;

			//dcost
			temp = sumon + 1.0 - 2*current;
			t1 = -lgamma(temp+1.0)-lgamma(model->m-temp+1.0);
			t1 -= -lgamma(sumon+1.0)-lgamma(model->m-sumon+1.0);
			t2 = 0.5*log((double)doc->total/2.0/PI);
			if (current == 1)	t2 = -t2;
			dcost = t1 + t2;

			//temp-learn new theta
			sumtheta = (double) sumon + 1.0 - 2*current;
			for (jj = 0; jj < model->m; jj++){
				if (v0[jj] == 1)
					theta1[jj] = 1.0/sumtheta;
				else
					theta1[jj] = 0.0;
			}
			variter = 0;
			prev_lkh = -1e50;
			sumtheta = 0.0;
			for (jj = 0; jj < model->m; jj++){
				sstheta[jj] = 0.0;
			}
			do{ // (can remove this loop) loop to update theta
				varlkh = 0.0;
				temp_varlkh = 0.0;

				for (n = 0; n < doc->length; n++){
					c = (double) doc->counts[n];
					w = doc->words[n];
					phisum = 0.0;
					temp_phisum = 0.0;
					for (jj = 0; jj < model->m; jj++){
						if (jj != j){
							if (v0[jj] == 1)
								phisum += theta0[jj]*model->beta0[jj][w];
						}else{
							if (current == 1)
								phisum += theta0[jj]*model->beta0[jj][w];
						}
						if (v0[jj] == 1)
							phi[jj] = theta1[jj]*model->beta0[jj][w];
						else
							phi[jj] = 0;
						temp_phisum += phi[jj];
					}
					varlkh += c*log(phisum);
					for (jj = 0; jj < model->m; jj++){
						temp = phi[jj];
						phi[jj] /= temp_phisum;
						cphi = c*phi[jj];
						if (phi[jj] > 0){
							temp_varlkh += cphi*(log(temp) - log(phi[jj]));
							sstheta[jj] += cphi;
							sumtheta += cphi;
						}
					}
				}
				//update theta
				for (jj = 0; jj < model->m; jj++){
					theta1[jj] = sstheta[jj]/sumtheta;
					sstheta[jj] = 0.0;
				}
				sumtheta = 0.0;
				conv = (prev_lkh - temp_varlkh)/prev_lkh;
				prev_lkh = temp_varlkh;
				variter ++;

			}while((variter < MAXITER) && ((conv > CONVERGED) || (conv < 0)));
			dl = temp_varlkh - varlkh;
			deltaq = dcost - dl;

			if (deltaq < 0){ //accept
				for (jj = 0; jj < model->m; jj++){
					theta0[jj] = theta1[jj];
				}
				sumon += 1 - 2*current;
				change ++;
			}else{ // reject
				v0[j] = current;
			}
		}
		iter ++;
	}while((iter < 10) && (change > 0));

}


void nullmodel(atd_corpus* corpus, atd_model* model, atd_ss* ss, atd_var* var){

	int d, j, n, w, argmaxlkh;
	double varlkh, c, temp, phisum;
	double normlkh, maxlkh, mininc;

	double* theta0 = malloc(sizeof(double)*model->m);
	double* theta1 = malloc(sizeof(double)*model->m);
	int* v0 = malloc(sizeof(int)*model->m);
	double * sstheta = malloc(sizeof(double)*model->m);
	double * phi = malloc(sizeof(double)*model->m);

	mininc = 1e50;

	for (d = 0; d < corpus->ndocs; d++){

		//compute lkh under all topics (only one present at a time)
		maxlkh = -1e100;
		for (j = 0; j < model->m; j++){
			theta0[j] = 0.0;
			v0[j] = 0;
			temp = 0.0;
			for (n = 0; n < corpus->docs[d].length; n++){
				c = (double) corpus->docs[d].counts[n];
				w = corpus->docs[d].words[n];
				temp += c*log(model->beta0[j][w]);
			}
			if (temp > maxlkh){
				maxlkh = temp;
				argmaxlkh = j;
			}
		}
		theta0[argmaxlkh] = 1.0;
		v0[argmaxlkh] = 1;

		update_null_theta(theta0, theta1, sstheta, v0, phi, model, &(corpus->docs[d]));

		for (j = 0; j < model->m; j++){
			model->theta0[j][d] = theta0[j];
			model->v0[j][d] = v0[j];
		}
		//compute lkh last time
		varlkh = 0.0;
		for (n = 0; n < corpus->docs[d].length; n++){
			c = (double) corpus->docs[d].counts[n];
			w = corpus->docs[d].words[n];
			phisum = 0.0;
			for (j = 0; j < model->m; j++){
				if (model->v0[j][d] == 0)
					continue;
				phisum += model->theta0[j][d]*model->beta0[j][w];
			}
			varlkh += c*log(phisum);

		}

		var->lkh0[d] = varlkh;

		normlkh = var->lkh0[d]/((double)corpus->docs[d].total);
		if (normlkh < mininc){
			mininc = normlkh;
			var->firstdoc = d;
		}

	}

	free(theta0);
	free(v0);
	free(theta1);
	free(sstheta);
	free(phi);
}




atd_model* new_atd_model(int ntopics, int nterms, int ndocs, char* model_name)
{
	int n, j, d;
	char filename[100];
	FILE* fileptr;
	float x;


	atd_model* model = malloc(sizeof(atd_model));
	model->D = ndocs;
    model->m = ntopics;
    model->n = nterms;
    model->zeta = 0.0;
    model->mu = 0.0;
    model->sumshared = 0.0;

    model->shared = malloc(sizeof(double)*nterms);
    model->betahat = malloc(sizeof(double)*nterms);
    for (n = 0; n < nterms; n++){
    	model->shared[n] = 0.0;
    	model->betahat[n] = 0.0;
    }
	model->beta = malloc(sizeof(double*)*(ntopics+1));
	model->beta0 = malloc(sizeof(double*)*ntopics);
	model->theta = malloc(sizeof(double*)*(ntopics+1));
	model->v = malloc(sizeof(int*)*(ntopics+1));
	model->theta0 = malloc(sizeof(double*)*ntopics);
	model->v0 = malloc(sizeof(int*)*ntopics);
	for (j = 0; j < (ntopics+1); j++){
		model->beta[j] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->beta[j][n] = 0.0;
		}
		model->theta[j] = malloc(sizeof(double)*ndocs);
		model->v[j] = malloc(sizeof(int)*ndocs);
		for (d = 0; d < ndocs; d++){
			model->theta[j][d] = 0.0;
			model->v[j][d] = 0;
		}
	}
	for (j = 0; j < ntopics; j++){
		model->beta0[j] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->beta0[j][n] = 0.0;
		}
		model->theta0[j] = malloc(sizeof(double)*ndocs);
		model->v0[j] = malloc(sizeof(double)*ndocs);
		for (d = 0; d < ndocs; d++){
			model->theta0[j][d] = 0.0;
			model->v0[j][d] = 0.0;
		}
	}

	sprintf(filename, "%s.beta", model_name);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    model->sumshared = 0.0;
    for (n = 0; n < nterms; n++){
		for (j = 0; j < ntopics; j++){
			fscanf(fileptr, " %f", &x);
			model->beta0[j][n] = exp(x);
			if (model->beta0[j][n] == 0)	model->beta0[j][n] = EPS;
			model->beta[j][n] = model->beta0[j][n];
		}
		fscanf(fileptr, " %f", &x);
		model->shared[n] = exp(x);
		if (model->shared[n] == 0)	model->shared[n] = EPS;
		model->sumshared += model->shared[n];
	}
    fclose(fileptr);

	model->u = malloc(sizeof(int)*nterms);
	for (n = 0; n < nterms; n++){
		model->u[n] = 0;
	}

    return(model);
}


atd_var * new_atd_var(int ndocs, int ntopics, int nmax, int nterms)
{
	int n, j, d;

    atd_var * var = malloc(sizeof(atd_var));

    var->phi = malloc(sizeof(double*)*nmax);
    for (n = 0; n < nmax; n++){
    	var->phi[n] = malloc(sizeof(double)*(ntopics+1));
    	for (j = 0; j < (ntopics+1); j++){
    		var->phi[n][j] = 0.0;
    	}
    }

	//var->sumphi = malloc(sizeof(double)*(ntopics+1));
    var->oldphi = malloc(sizeof(double)*(ntopics+1));
	var->temp_betass0 = malloc(sizeof(double)*(ntopics));
	var->temp_sumbetass0 = malloc(sizeof(double)*(ntopics));
	for (j = 0; j < (ntopics+1); j++){
		//var->sumphi[j] = 0.0;
		var->oldphi[j] = 0.0;
		if (j < ntopics){
			var->temp_betass0[j] = 0.0;
			var->temp_sumbetass0[j] = 0.0;
		}
	}

	var->lkh0 = malloc(sizeof(double)*ndocs);
	var->sdocs = malloc(sizeof(int)*ndocs);
	var->docchk = malloc(sizeof(int)*ndocs);
	var->lkh = malloc(sizeof(double)*ndocs);
	var->lkhratio = malloc(sizeof(double)*ndocs);
	for (d = 0; d < ndocs; d++){
		var->lkh0[d] = 0.0;
		var->sdocs[d] = 0;
		var->docchk[d] = 0;
		var->lkh[d] = 0.0;
		var->lkhratio[d] = 0.0;
	}
	var->slen = 0;
	var->firstdoc = 0;
	var->swrdlen = 0;
	var->wrdchk = malloc(sizeof(int)*nterms);
	var->swrds = malloc(sizeof(int)*nterms);
	var->tempwrdchk = malloc(sizeof(int)*nterms);
	for (n = 0; n < nmax; n++){
		var->wrdchk[n] = 0;
		var->swrds[n] = 0;
		var->tempwrdchk[n] = 0;
	}


	return(var);
}

atd_ss * new_atd_ss(atd_model* model)
{
	int j, n, d;
	atd_ss * ss;
    ss = malloc(sizeof(atd_ss));
	ss->beta = malloc(sizeof(double)*model->n);
	ss->sum_beta = 0.0;
	for (n = 0; n < model->n; n++){
		ss->beta[n] = 0.0;
	}
	ss->beta0 = malloc(sizeof(double*)*model->m);
	ss->sum_beta0 = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		ss->beta0[j] = malloc(sizeof(double)*model->n);
		ss->sum_beta0[j] = 0.0;
		for (n = 0; n < model->n; n++){
			ss->beta0[j][n] = 0.0;
		}
	}
	ss->theta = malloc(sizeof(double*)*(model->m+1));
	for (j = 0; j < (model->m+1); j++){
		ss->theta[j] = malloc(sizeof(double)*model->D);
		for (d = 0; d < model->D; d++){
			ss->theta[j][d] = 0.0;
		}
	}
    return(ss);
}


atd_corpus* read_data(const char* data_filename, int ntopics)
{
	int OFFSET = 0;
    FILE *fileptr;
    int length, count, word, n, nd, nw, corpus_total = 0;
    double norm = 0.0;
    //int j;
    atd_corpus* c;

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(atd_corpus));
    fileptr = fopen(data_filename, "r");
    nd = 0; nw = 0;
    c->docs = malloc(sizeof(document) * 1);
    while ((fscanf(fileptr, "%10d", &length) != EOF))
    {
	c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
	c->docs[nd].tpcont = 0.0;
	c->docs[nd].length = length;
	c->docs[nd].total = 0;
	c->docs[nd].words = malloc(sizeof(int)*length);
	c->docs[nd].counts = malloc(sizeof(int)*length);
	c->docs[nd].normcnts = malloc(sizeof(double)*length);
	norm = 0.0;
	for (n = 0; n < length; n++)
	{
	    fscanf(fileptr, "%10d:%10d", &word, &count);
	    word = word - OFFSET;
	    c->docs[nd].words[n] = word;
	    c->docs[nd].counts[n] = count;
	    c->docs[nd].total += count;
	    if (word >= nw) { nw = word + 1; }
	    norm += ((double) count)*((double) count);
	}
	norm = sqrt(norm);
	for (n = 0; n < length; n++){
		c->docs[nd].normcnts[n] = ((double) c->docs[nd].counts[n])/norm;
	}

	corpus_total += c->docs[nd].total;
        nd++;
    }
    fclose(fileptr);
    c->ndocs = nd;
    c->nterms = nw;
    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);
    printf("total             : %d\n", corpus_total);
    return(c);
}

int max_corpus_length(atd_corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->ndocs; n++)
	if (c->docs[n].length > max) {
		max = c->docs[n].length;
		//printf("doc %d, length = %d\n",n, c->docs[n].length);
	}
    return(max);
}

void write_atd_model(atd_model * model, char * root, atd_corpus * corpus, atd_var* var){

	char filename[200];
    FILE* fp;
    int n, j, d, dd, nn;

	sprintf(filename, "%s/test.sdocs", root);
	fp = fopen(filename, "w");
	for (dd = 0; dd < var->slen; dd++){
		d = var->sdocs[dd];
		fprintf(fp,"%s, %d\n",corpus->docs[d].title, d);
	}
	fclose(fp);

	sprintf(filename, "%s/test.theta", root);
	fp = fopen(filename, "w");
	for (dd = 0; dd < var->slen; dd++){
		d = var->sdocs[dd];
		for (j = 0; j <= model->m; j++){
			if (j < model->m)
				fprintf(fp,"%lf ",model->theta[j][d]);
			else
				fprintf(fp,"%lf",model->theta[j][d]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);

	sprintf(filename, "%s/test.beta", root);
	fp = fopen(filename, "w");
	for (n = 0; n < model->n; n++){
		fprintf(fp,"%5.10lf %d %d\n", model->beta[model->m][n], model->u[n], var->wrdchk[n]);
	}
	fclose(fp);

	sprintf(filename, "%s/test.u", root);
	fp = fopen(filename, "w");
	for (nn = 0; nn < var->swrdlen; nn++){
		n = var->swrds[nn];
		fprintf(fp,"%d, %5.10lf, %d\n", n, model->beta[model->m][n],model->u[n]);
	}
	fclose(fp);
}

atd_trmodel* new_atd_trmodel(int ntopics, int nterms, int ndocs)
{
	int n, j, d;

	atd_trmodel* model = malloc(sizeof(atd_trmodel));
	model->D = ndocs;
    model->m = ntopics;
    model->n = nterms;

	model->beta = malloc(sizeof(double*)*ntopics);
	model->theta = malloc(sizeof(double*)*ntopics);
	model->v = malloc(sizeof(int*)*ntopics);
	for (j = 0; j < ntopics; j++){
		model->beta[j] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->beta[j][n] = 0.0;
		}
		model->theta[j] = malloc(sizeof(double)*ndocs);
		model->v[j] = malloc(sizeof(int)*ndocs);
		for (d = 0; d < ndocs; d++){
			model->theta[j][d] = 0.0;
			model->v[j][d] = 0;
		}
	}

    return(model);
}


atd_trmodel* load_model(char* model_root, int ndocs){

	char filename[100];
	FILE* fileptr;
	int j, n, num_topics, num_terms, num_docs;
	float x;

	atd_trmodel* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fclose(fileptr);

	model  = new_atd_trmodel(num_topics, num_terms, ndocs);
	model->n = num_terms;
	model->m = num_topics;
	model->D = ndocs;

	sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    for (n = 0; n < num_terms; n++){
		for (j = 0; j < num_topics; j++){
			fscanf(fileptr, " %f", &x);
			model->beta[j][n] = exp(x);
			if (model->beta[j][n] == 0) model->beta[j][n] = EPS;
		}
		fscanf(fileptr, " %f", &x);
	}
    fclose(fileptr);

    return(model);

}
