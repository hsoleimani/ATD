#include "opt.h"
//static gsl_vector theta_evaluate(void *instance, const gsl_vector *x,
//		gsl_vector *grad, const int n, const gsl_vector step);
double my_f (const gsl_vector *v, void *params);
void my_df (const gsl_vector *v, void *params, gsl_vector *df);
void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df);


void optimize_alpha(gsl_vector * x, void * data, int n, gsl_vector * x2){

	size_t iter = 0;
	int status, j;

	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer *s;

	gsl_multimin_function_fdf my_func;

	my_func.n = n;
	my_func.f = my_f;
	my_func.df = my_df;
	my_func.fdf = my_fdf;
	my_func.params = data;

	T = gsl_multimin_fdfminimizer_conjugate_fr;
	//T = gsl_multimin_fdfminimizer_vector_bfgs2;
	s = gsl_multimin_fdfminimizer_alloc (T, n);

	//gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.1, 0.01);
	gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-4);
	//printf("%lf\n",-s->f);
	do
	{
	  iter++;
	  status = gsl_multimin_fdfminimizer_iterate (s);
	  //printf ("status = %s\n", gsl_strerror (status));
	  if (status){
		  if (iter == 1){
			  for (j = 0; j < n; j++){
				  gsl_vector_set(x2, j, gsl_vector_get(x, j));
			  }
		  }
		break;
	  }

	  status = gsl_multimin_test_gradient (s->gradient, 1e-3);
	  //print_state (iter, s);
	  //printf ("status = %s\n", gsl_strerror (status));
	  /*if (status == GSL_SUCCESS)
		printf ("Minimum found at:\n");

	  printf ("%5d %.5f %.5f %10.5f\n", iter,
			  gsl_vector_get (s->x, 0),
			  gsl_vector_get (s->x, 1),
			  s->f);*/
	  for (j = 0; j < n; j++){
		  gsl_vector_set(x2, j, gsl_vector_get(s->x, j));
	  }

	}
	while (status == GSL_CONTINUE && iter < 100);
	//printf("%lf\n",-s->f);
	gsl_multimin_fdfminimizer_free (s);
	//gsl_vector_free (x);

	//return 0;
}


double my_f (const gsl_vector *v, void *params)
{

	M4_alphaopt * alphaopt=(M4_alphaopt *) params;
	double f = 0.0;
	int j, ind, t, g;


	for (t = 0; t < alphaopt->ngenres; t++){
		alphaopt->sumalpha[t] = 0.0;
		alphaopt->sumlgalpha[t] = 0.0;
		for (j = 0; j < alphaopt->ntopics; j++){
			ind = t*alphaopt->ntopics + j;
			 alphaopt->alpha[t][j] = exp(gsl_vector_get(v, ind));
			 if (alphaopt->alpha[t][j] == 0) alphaopt->alpha[t][j] = EPS;
			 alphaopt->lgalpha[t][j] = lgamma(alphaopt->alpha[t][j]);
			 alphaopt->sumalpha[t] += alphaopt->alpha[t][j];

			 alphaopt->sumlgalpha[t] += alphaopt->lgalpha[t][j];
		}
	}

	f = 0.0;
	for (g = 0; g < alphaopt->ngroups; g++){
		for (t = 0; t < alphaopt->ngenres; t++){
			if (alphaopt->mu[g][t] == 0)
				continue;
			f += alphaopt->mu[g][t]*((double)alphaopt->ngdocs[g])*
					(lgamma(alphaopt->sumalpha[t]) - alphaopt->sumlgalpha[t]);
			for (j = 0; j < alphaopt->ntopics; j++){
				f += alphaopt->mu[g][t]*(alphaopt->alpha[t][j]-1)*alphaopt->alphass[t][j][g];
			}
		}
	}
	f = -f;

	return(f);
}



void my_df (const gsl_vector *v, void *params, gsl_vector *df)
{

	M4_alphaopt * alphaopt=(M4_alphaopt *) params;
	//double f = 0.0;
	int j, ind, t, g;


	for (t = 0; t < alphaopt->ngenres; t++){
		alphaopt->sumalpha[t] = 0.0;
		alphaopt->sumlgalpha[t] = 0.0;
		for (j = 0; j < alphaopt->ntopics; j++){
			ind = t*alphaopt->ntopics + j;
			 alphaopt->alpha[t][j] = exp(gsl_vector_get(v, ind));
			 if (alphaopt->alpha[t][j] == 0) alphaopt->alpha[t][j] = EPS;
			 alphaopt->lgalpha[t][j] = lgamma(alphaopt->alpha[t][j]);
			 alphaopt->sumalpha[t] += alphaopt->alpha[t][j];

			 alphaopt->sumlgalpha[t] += alphaopt->lgalpha[t][j];
			 alphaopt->grad[ind] = 0.0;
		}
	}

	//f = 0.0;
	for (g = 0; g < alphaopt->ngroups; g++){
		for (t = 0; t < alphaopt->ngenres; t++){
			if (alphaopt->mu[g][t] == 0)
				continue;
			//f += alphaopt->mu[g][t]*((double)alphaopt->ngdocs[g])*
			//		(lgamma(alphaopt->sumalpha[t]) - alphaopt->sumlgalpha[t]);
			for (j = 0; j < alphaopt->ntopics; j++){
				ind = t*alphaopt->ntopics + j;
				alphaopt->grad[ind] += alphaopt->mu[g][t]*((double)alphaopt->ngdocs[g])*
						(gsl_sf_psi(alphaopt->sumalpha[t]) - gsl_sf_psi(alphaopt->alpha[t][j]));
				alphaopt->grad[ind] += alphaopt->mu[g][t]*alphaopt->alphass[t][j][g];
				//f += alphaopt->mu[g][t]*(alphaopt->alpha[t][j]-1)*alphaopt->alphass[t][j][g];
			}
		}
	}
	//f = -f;

	for (t = 0; t < alphaopt->ngenres; t++){
		for (j = 0; j < alphaopt->ntopics; j++){
			ind = t*alphaopt->ntopics + j;
			alphaopt->grad[ind] *= -alphaopt->alpha[t][j];
			gsl_vector_set(df, ind, alphaopt->grad[ind]);
		}
	}

}


void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df)
{


	M4_alphaopt * alphaopt=(M4_alphaopt *) params;
	int j, ind, t, g;


	for (t = 0; t < alphaopt->ngenres; t++){
		alphaopt->sumalpha[t] = 0.0;
		alphaopt->sumlgalpha[t] = 0.0;
		for (j = 0; j < alphaopt->ntopics; j++){
			ind = t*alphaopt->ntopics + j;

			alphaopt->alpha[t][j] = exp(gsl_vector_get(v, ind));
			if (alphaopt->alpha[t][j] == 0) alphaopt->alpha[t][j] = EPS;
			alphaopt->lgalpha[t][j] = lgamma(alphaopt->alpha[t][j]);
			alphaopt->sumalpha[t] += alphaopt->alpha[t][j];

			alphaopt->sumlgalpha[t] += alphaopt->lgalpha[t][j];
			alphaopt->grad[ind] = 0.0;
		}
	}

	*f = 0.0;
	for (g = 0; g < alphaopt->ngroups; g++){
		for (t = 0; t < alphaopt->ngenres; t++){
			if (alphaopt->mu[g][t] == 0)
				continue;
			*f += alphaopt->mu[g][t]*((double)alphaopt->ngdocs[g])*
					(lgamma(alphaopt->sumalpha[t]) - alphaopt->sumlgalpha[t]);
			for (j = 0; j < alphaopt->ntopics; j++){
				ind = t*alphaopt->ntopics + j;
				alphaopt->grad[ind] += alphaopt->mu[g][t]*((double)alphaopt->ngdocs[g])*
						(gsl_sf_psi(alphaopt->sumalpha[t]) - gsl_sf_psi(alphaopt->alpha[t][j]));
				alphaopt->grad[ind] += alphaopt->mu[g][t]*alphaopt->alphass[t][j][g];
				*f += alphaopt->mu[g][t]*(alphaopt->alpha[t][j]-1)*alphaopt->alphass[t][j][g];
			}
		}
	}
	*f = -(*f);

	for (t = 0; t < alphaopt->ngenres; t++){
		for (j = 0; j < alphaopt->ntopics; j++){
			ind = t*alphaopt->ntopics + j;
			alphaopt->grad[ind] *= -alphaopt->alpha[t][j];
			gsl_vector_set(df, ind, alphaopt->grad[ind]);
		}
	}

}
