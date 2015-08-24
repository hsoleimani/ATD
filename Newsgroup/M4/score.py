#import os
#import numpy as np
#from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from scipy.special import gammaln
from scipy.special import digamma
from sklearn import metrics

def compute_score(seed, path, tlblfile, tgroupfile,resfile, anomlist):
	B = 2000
	# read mu
	mu = np.exp(np.loadtxt(path+'/M4/dirM4/testfinal.mu'))
	G = mu.shape[0]
	T = mu.shape[1]
	mu = mu/np.sum(mu,1).reshape(-1,1)
	# read gamma
	fname = path+'/M4/dirM4/testfinal.gamma'+str(0)
	gamma0 = np.loadtxt(fname)
	D = gamma0.shape[0]
	M = gamma0.shape[1]
	gamma = np.zeros((D,M,T))
	gamma[:,:,0] = gamma0
	for t in range(1,T):
		fname = path+'/M4/dirM4/testfinal.gamma'+str(t)
		gamma[:,:,t] = np.loadtxt(fname)

	# read group memberships
	groups = np.loadtxt(tgroupfile,dtype=int)
	docscore = np.zeros(D)

	def log_sum(a,b):
	  if (a < b):
		  v = b + np.log(1 + np.exp(a-b))
	  else:
		  v = a + np.log(1 + np.exp(b-a))
	  return(v)

	logmu = np.log(mu+1e-100)
	for d in range(D):
		g = groups[d]
		tempscore = np.zeros(B)
		# compute constant terms in lnpdf:	
		const_terms = gammaln(np.sum(gamma[d,:,:],0))-np.sum(gammaln(gamma[d,:,:]),0)

		if np.any(mu[g,:]==1): # can exactly compute score (entropy of a dirichlet distn)
			t = np.where(mu[g,:]==1)[0][0]
			sumgamma = np.sum(gamma[d,:,t])
			docscore[d] = (sumgamma-M)*digamma(sumgamma)-np.sum((gamma[d,:,t]-1.0)*digamma(gamma[d,:,t]))-const_terms[t]
		else: # need Monte Carlo
			for b in range(B):
				t = np.random.choice(T, 1, p = mu[g,:])[0]
				theta = np.random.dirichlet(gamma[d,:,t])
				temp = (gamma[d,:,:]-1.0)*np.tile(np.log(theta+1e-200).reshape(-1,1),(1,T))
				lnpdf = np.sum(temp,0) + const_terms;
				q = 0
				for t0 in range(T):
					if t0 > 0:
						q = log_sum(q, logmu[g,t0]+lnpdf[t0])
					else:
						q = logmu[g,t0]+lnpdf[t0]
				tempscore[b] += q
			docscore[d] = -np.mean(tempscore)

	# compute score of groups
	groupscore = np.zeros(G)
	for g in range(G):
		ind = np.where(groups == g)
		groupscore[g] = np.mean(docscore[ind])

	# compute recall/precision

	highscores = np.argsort(-groupscore)
	fp = open(tlblfile)
	lbllist = fp.readlines()
	fp.close()
	#fpres = open('results_'+str(G)+'_'+str(T)+'.txt','w')
	fpres = open(resfile,'a')
	f1scores = list()

	flist = list()
	reclist = list()
	preclist = list()
	auclist = list()

	for g in highscores[0:2]:
		ind = np.where(groups == g)[0]
		Ssize = len(ind)

		step_num = np.zeros(len(anomlist))
		step_recall = np.zeros(len(anomlist))
		step_precision = np.zeros(len(anomlist))
		step_F1 = np.zeros(len(anomlist))
		step_auc = np.zeros(len(anomlist))

		for a0,anomlbl in enumerate(anomlist):
			a = [1 for x in lbllist if anomlbl in x]
			Danom = float(len(a))
			rec = np.zeros(Ssize)
			prec = np.zeros(Ssize)
			tp = 0
			for i in range(Ssize):
				doclbl = lbllist[ind[i]]
				if anomlbl in doclbl:
					tp += 1.0
				prec[i] = (tp/(i+1.0))
				rec[i] = (tp/Danom)
			#fpres.write('group %d: size = %d, anomscore = %f, anomlbl = %s, recall = %f, precision = %f\n' %(g,Ssize,anomcnt[g],anomlbl,rec,prec))
			step_num[a0] = tp
			step_recall[a0] = rec[-1]
			step_precision[a0] = prec[-1]
			if (rec[-1]+prec[-1]) > 0:
				step_F1[a0] = 2.0*prec[-1]*rec[-1]/(rec[-1]+prec[-1])
			if Ssize < 2:
				step_auc[a0] = 0.0
			else:
				step_auc[a0] = metrics.auc(rec, prec)

		a0 = np.argmax(step_num)
		flist.append(step_F1[a0])
		reclist.append(step_recall[a0])
		preclist.append(step_precision[a0])
		auclist.append(step_auc[a0])
		fpres.write('group %d: size = %d, anomscore = %f, anomlbl = %s, recall = %f, precision = %f, auc = %f, f1 = %f\n' %(g,Ssize,groupscore[g],anomlist[a0],step_recall[a0],step_precision[a0],step_auc[a0],step_F1[a0]))

	f1scores.append(np.mean(flist))

	fpres.close()
	return(f1scores)

