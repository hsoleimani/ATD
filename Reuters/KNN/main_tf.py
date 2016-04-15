import os, re
import numpy as np
from sklearn import svm
from sklearn import metrics
from scipy.sparse import csr_matrix

LDAPath = '../../lda'
path = '../'

anom_per = 50

seed0 = 3181914101
np.random.seed(seed0)
N = 9469
trainfile = path + '/data/trdocs.txt'
testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
tlblfile = path + '/data/tlbls'+str(anom_per) + '.txt'


anomlist = ['coffee','ship']
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
Mlist = np.arange(14,32,2)
F1score = np.zeros((len(Mlist), len(Klist)))
fpres = open('results_tf.txt','w')
fpres.write('')
fpres.close()
for m1,M in enumerate(Mlist):
	# run lda on training data
	seed = np.random.randint(seed0)
	cmdtxt = LDAPath + '/lda ' + str(seed) + ' est 0.1 ' + str(M) + ' ' + LDAPath + '/settings2.txt ' + trainfile + ' seeded dirlda' 
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
	cmdtxt = LDAPath + '/lda ' + str(seed) + ' inf ' + LDAPath + '/settings2.txt dirlda/final ' + testfile + ' dirlda/test' 
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
		temp = trX.dot(trX.T).toarray()
		for d in range(Dt):
			dist = 1.0 - temp[d,:]#np.dot(trX,tX[d,:])
			dist[d] = 1.0
			Rstr[d] = np.sort(dist)[K]
		
		# K-NN on test set
		pval = np.zeros(Dt)
		temp = tX.dot(trX.T).toarray()
		for d in range(Dt):
			dist = 1.0 - temp[d,:]#np.dot(trX,tX[d,:])
			pval[d] = np.mean(np.sort(dist)[K] < Rstr)
			
		# compute score of each group
		anomcnt = np.zeros(M)
		for g in range(M):
			ind = np.where(group_assgnmt == g)[0]
			anomcnt[g] = np.mean(pval[ind])

		maxgroupind = np.argsort(anomcnt)
		flist = list()
		reclist = list()
		preclist = list()
		auclist = list()

		fpres = open('results_tf.txt','a')
		fpres.write('************************************\n')
		for g in maxgroupind[0:2]:

			ind = np.where(group_assgnmt == g)[0]
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
				l = 1
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
					step_auc[a0] = metrics.auc(rec, prec)

			# assign this group to the lbl of argmax(step_num)
			a0 = np.argmax(step_num)
			flist.append(step_F1[a0])
			reclist.append(step_recall[a0])
			preclist.append(step_precision[a0])
			auclist.append(step_auc[a0])
			fpres.write('group %d: size = %d, anomscore = %f, anomlbl = %s, recall = %f, precision = %f, auc = %f, f1 = %f\n' %(g,Ssize,anomcnt[g],anomlist[a0],step_recall[a0],step_precision[a0],step_auc[a0],step_F1[a0]))
			
		F1score[m1,n1] = np.mean(flist)
		print('M = %d, K = %d, F1-score = %f' %(M, K, F1score[m1,n1]))
		fpres.write('M = %d, K = %d, F1-score = %f\n' %(M, K, F1score[m1,n1]))
		fpres.close()

fpres = open('results_tf.txt','a')
fpres.write('############################################\n')
ind = np.unravel_index(F1score.argmax(), F1score.shape)
fpres.write('Best F1-score: M = %d, K = %d, F1-score = %f' %(Mlist[ind[0]],Klist[ind[1]],F1score[ind[0],ind[1]]))
fpres.close()
os.system('rm -r dirlda')
