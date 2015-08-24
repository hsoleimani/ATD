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
# count total number of anomalies
Danom = 0.0;
for a0,anomlbl in enumerate(anomlist):
	a = [1 for x in lbllist if anomlbl in x]
	Danom += float(len(a))


Klist = np.arange(2,30,2)
Mlist = [6,8,10,12,14,16,18,20]
F1score = np.zeros((len(Mlist), len(Klist)))
fpres = open('results_indv.txt','w')
fpres.write('')
fpres.close()
for m1,M in enumerate(Mlist):
	# run lda on training data
	seed = np.random.randint(seed0)
	cmdtxt = LDAPath + '/lda ' + str(seed) + ' est 0.1 ' + str(M) + ' ' + LDAPath + '/settings.txt ' + trainfile + ' seeded dirlda' 
	print('Running LDA on training set')
	os.system(cmdtxt + ' > /dev/null')

	# read topic proportions
	train_theta = np.loadtxt('dirlda/final.gamma')
	sumtheta = np.sum(train_theta,1)
	train_theta = train_theta/sumtheta.reshape(-1,1)
	normtrain = np.sqrt(np.sum(train_theta**2,1))
	Dtr = len(normtrain)

	# run lda on test set
	seed = np.random.randint(seed0)
	cmdtxt = LDAPath + '/lda ' + str(seed) + ' inf ' + LDAPath + '/settings.txt dirlda/final ' + testfile + ' dirlda/test' 
	print('Running LDA on the test set')
	os.system(cmdtxt + ' > /dev/null')

	# read topic proportions
	test_theta = np.loadtxt('dirlda/test-gamma.dat')
	sumtheta = np.sum(test_theta,1)
	test_theta = test_theta/sumtheta.reshape(-1,1)
	normtest = np.sqrt(np.sum(test_theta**2,1))
	Dt = len(normtest)

	# hard assign each doc to a topic to form the groups
	group_assgnmt = np.argmax(test_theta,1)

	for n1,K in enumerate(Klist):

		# K-NN on training set
		Rstr = np.zeros(Dtr)
		for d in range(Dtr):
			dist = 1.0 - np.dot(train_theta,train_theta[d,:])/normtrain/normtrain[d]
			dist[d] = 1.0
			Rstr[d] = np.sort(dist)[K]
		
		# K-NN on test set
		pval = np.zeros(Dt)
		for d in range(Dt):
			dist = 1.0 - np.dot(train_theta,test_theta[d,:])/normtrain/normtest[d]
			pval[d] = np.mean(np.sort(dist)[K] < Rstr)
			

		fpres = open('results_indv.txt','a')

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
		step_F1 = 2.0*recall[-1]*precision[-1]/(recall[-1]+precision[-1])


		fpres.write('recall = %f, precision = %f, auc = %f, f1 = %f\n' %(recall[-1],precision[-1],step_auc,step_F1))
			
		F1score[m1,n1] = step_F1
		print('M = %d, K = %d, F1-score = %f' %(M, K, F1score[m1,n1]))
		fpres.write('M = %d, K = %d, F1-score = %f\n' %(M, K, F1score[m1,n1]))
		fpres.close()

fpres = open('results_indv.txt','a')
fpres.write('############################################\n')
amax = np.argmax(F1score)
m1 = amax/len(Klist)
n1 = amax%len(Klist)
fpres.write('Best F1-score: M = %d, K = %d, F1-score = %f' %(Mlist[m1],Klist[n1],F1score[m1,n1]))
fpres.close()
