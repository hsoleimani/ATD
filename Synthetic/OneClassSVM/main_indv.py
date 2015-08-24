import os, re
import numpy as np
from sklearn import svm
from sklearn import metrics

LDAPath = '/home/studadmin/Dropbox/ATDFinal/lda'
path = '/home/studadmin/Dropbox/ATDFinal/Synthetic'

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
F1score = np.zeros(len(nulist))
fpres = open('results_indv.txt','w')
fpres.write('')
fpres.close()
for n1,nu in enumerate(nulist):
	# train svm
	clf = svm.OneClassSVM(nu=nu, kernel="linear")
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
	nu_F1 = 2.0*recall[-1]*precision[-1]/(recall[-1]+precision[-1])

	fpres = open('results_indv.txt','a')
	fpres.write('recall = %f, precision = %f, auc = %f, f1 = %f\n' %(recall[-1],precision[-1],nu_auc,nu_F1))
	fpres.close()

	F1score[n1] = nu_F1
	print('nu = %f, AUC = %f' %(nu, nu_auc))


fpres = open('results_indv.txt','a')
fpres.write('############################################\n')
amax = np.argmax(F1score)
fpres.write('Best F1-score: nu = %f, F1-score = %f' %(nulist[amax],F1score[amax]))
fpres.close()
