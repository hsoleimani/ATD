import os, re
import numpy as np
from sklearn import svm
from sklearn import metrics
from scipy.sparse import csr_matrix

LDAPath = '../../lda'
path = '../'

anom_per = 20

seed0 = 3181914101
np.random.seed(seed0)
N = 33565
trainfile = path + '/data/trdocs.txt'
testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
tlblfile = path + '/data/tlbls'+str(anom_per) + '.txt'

TopN = 215


anomlist = ['talk.politics.mideast','rec.sport.hockey']
fp = open(tlblfile)
lbllist = fp.readlines()
fp.close()
# count total number of anomalies
Danom = 0.0;
for a0,anomlbl in enumerate(anomlist):
	a = [1 for x in lbllist if anomlbl in x]
	Danom += float(len(a))

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
		trX[d,int(w)] = float(cnts[n])
	trX[d,:] /= np.sqrt(np.sum(trX[d,:]**2))
trX = csr_matrix(trX)

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
		tX[d,int(w)] = float(cnts[n])
	tX[d,:] /= np.sqrt(np.sum(tX[d,:]**2))
tX = csr_matrix(tX)

Klist = np.hstack((1,np.arange(2,30,2)))
F1score = np.zeros(len(Klist))
fpres = open('results_indv_tf.txt','w')
fpres.write('')
fpres.close()

for n1,K in enumerate(Klist):

	# K-NN on training set
	Rstr = np.zeros(Dtr)
	temp = trX.dot(trX.T).toarray()
	for d in range(Dtr):
		dist = 1.0 - temp[d,:]#np.dot(trX,trX[d,:])
		dist[d] = 1.0
		Rstr[d] = np.sort(dist)[K]
	
	# K-NN on test set
	pval = np.zeros(Dt)
	temp = tX.dot(trX.T).toarray()
	for d in range(Dt):
		dist = 1.0 - temp[d,:]#np.dot(trX,tX[d,:])
		pval[d] = np.mean(np.sort(dist)[K] < Rstr)
		

	fpres = open('results_indv_tf.txt','a')

	anom_sorted = np.argsort(pval)

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

	step_auc = metrics.auc(recall, precision)
	step_F1 = 0.0
	if (recall[-1]+precision[-1]) > 0:
		step_F1 = 2.0*recall[-1]*precision[-1]/(recall[-1]+precision[-1])



	fpres.write('recall = %f, precision = %f, auc = %f, f1 = %f\n' %(recall[-1],precision[-1],step_auc,step_F1))
		
	F1score[n1] = step_F1
	print('K = %d, F1-score = %f' %(K, F1score[n1]))
	fpres.write('K = %d, F1-score = %f\n' %(K, F1score[n1]))
	fpres.close()

fpres = open('results_indv_tf.txt','a')
fpres.write('############################################\n')
amax = np.argmax(F1score)
fpres.write('Best F1-score: K = %d, F1-score = %f' %(Klist[amax],F1score[amax]))
fpres.close()
