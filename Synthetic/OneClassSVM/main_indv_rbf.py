import os, re
import numpy as np
from sklearn import svm
from sklearn import metrics
from numpy import unravel_index

LDAPath = '../../lda'
path = '../'

anom_per = 100
M = 10
seed0 = 3181914101
np.random.seed(seed0)

TopN = 64

N = 2998
trainfile = path + '/data/trdocs.txt'
testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
tlblfile = path + '/data/tlbls'+str(anom_per) + '.txt'


anomlist = ['11','10']
fp = open(tlblfile)
lbllist = fp.readlines()
fp.close()

# read training docs
fp = open(trainfile,'r')
rawdocs = fp.readlines()
fp.close()
Dtr = len(rawdocs)

trX = np.zeros((Dtr,N))
for d,doc in enumerate(rawdocs):
	wrds = re.findall('([0-9]*):[0-9]*',doc)
	cnts = re.findall('[0-9]*:([0-9]*)',doc)
	total = np.sum([float(x) for x in cnts])
	for n,w in enumerate(wrds):
		#trX[d,int(w)] = 1.0
		trX[d,int(w)] = float(cnts[n])/total

# read test docs
fp = open(testfile,'r')
rawdocs = fp.readlines()
fp.close()
Dt = len(rawdocs)

tX = np.zeros((Dt,N))
for d,doc in enumerate(rawdocs):
	wrds = re.findall('([0-9]*):[0-9]*',doc)
	cnts = re.findall('[0-9]*:([0-9]*)',doc)
	total = np.sum([float(x) for x in cnts])
	for n,w in enumerate(wrds):
		#tX[d,int(w)] = 1.0
		tX[d,int(w)] = float(cnts[n])/total

# count total number of anomalies
Danom = 0.0;
for a0,anomlbl in enumerate(anomlist):
	a = [1 for x in lbllist if anomlbl in x]
	Danom += float(len(a))


nulist = np.arange(1e-5, 0.4, 0.05)
gammalist = np.logspace(1.7, 4, 15)
F1score = np.zeros((len(nulist), len(gammalist)))
fpres = open('results_rbf_indv.txt','w')
fpres.write('')
fpres.close()
for n1,nu in enumerate(nulist):

	for g1, gamma in enumerate(gammalist):

		# train svm
		clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
		clf.fit(trX)

		# test svm
		#pred_test = clf.predict(tX)
		anom_score = clf.decision_function(tX)[:,0]

		anom_sorted = np.argsort(anom_score)

		# compute rec, prec
		recall = np.zeros(TopN)
		precision = np.zeros(TopN)
	
		tp = 0.0
		for i,ind in enumerate(anom_sorted[0:TopN]): 
			doclbl = lbllist[ind]
			anomchk = 0
			for a0,anomlbl in enumerate(anomlist):
				if anomlbl in doclbl:
					anomchk = 1
					break
			if anomchk == 1:
				tp += 1.0
			precision[i] = (tp/(i+1.0))
			recall[i] = (tp/Danom)

		nu_auc = metrics.auc(recall, precision)
		if (recall[-1]+precision[-1]) > 0:
			nu_F1 = 2.0*recall[-1]*precision[-1]/(recall[-1]+precision[-1])
		else:
			nu_F1 = 0.0

		fpres = open('results_rbf_indv.txt','a')
		fpres.write('nu = %f, gamma = %f, recall = %f, precision = %f, auc = %f, f1 = %f\n' %(nu, gamma, recall[-1],precision[-1],nu_auc,nu_F1))
		fpres.close()

		F1score[n1,g1] = nu_F1
		print('nu = %f, gamma = %f, AUC = %f' %(nu,gamma, nu_auc))


fpres = open('results_rbf_indv.txt','a')
fpres.write('############################################\n')
ind = unravel_index(F1score.argmax(), F1score.shape)
fpres.write('Best F1-score: nu = %f, gamma = %f, F1-score = %f' %(nulist[ind[0]],gammalist[ind[1]],F1score[ind[0],ind[1]]))
fpres.close()
