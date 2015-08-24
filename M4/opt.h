/*
 * opt.h
 *
 *  Created on: Mar 18, 2015
 *      Author: studadmin
 */

#ifndef OPT_H_
#define OPT_H_

#include "M4.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <gsl/gsl_multimin.h>


void optimize_alpha(gsl_vector * x, void * data, int n, gsl_vector * xx);



#endif /* OPT_H_ */

