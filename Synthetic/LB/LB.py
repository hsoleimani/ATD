import os, re
import numpy as np
from sklearn import metrics

LDAPath = '/home/studadmin/Dropbox/ATDFinal/lda'
path = '/home/studadmin/Dropbox/ATDFinal/Synthetic'

anom_per = 100
seed0 = 3181914101
np.random.seed(seed0)

N = 2998
trainfile = path + '/data/trdocs.txt'
testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
tlblfile = path + '/data/tlbls'+str(anom_per) + '.txt'


anomlist = ['11','10']
fp = open(tlblfile)
lbllist = fp.readlines()
fp.close()

lkh0 = np.loadtxt(path + '/dirtest1/test.normlkh0')


Mlist = [6,8,10,12,14,16,18,20]
F1score = np.zeros(len(Mlist))
fpres = open('results.txt','w')
fpres.write('')
fpres.close()
for m1,M in enumerate(Mlist):
	# run lda on test data
	seed = np.random.randint(seed0)
	cmdtxt = LDAPath + '/lda ' + str(seed) + ' est 0.1 ' + str(M) + ' ' + LDAPath + '/settings.txt ' + testfile + ' seeded dirlda' 
	print('Running LDA on test set')
	os.system(cmdtxt + ' > /dev/null')

	# read topic proportions
	test_theta = np.loadtxt('dirlda/final.gamma')
	sumtheta = np.sum(test_theta,1)
	test_theta = test_theta/sumtheta.reshape(-1,1)
	normtest = np.sqrt(np.sum(test_theta**2,1))
	Dt = len(normtest)

	# hard assign each doc to a topic to form the groups
	group_assgnmt = np.argmax(test_theta,1)

	anomcnt = np.zeros(M)
	for g in range(M):
		ind = np.where(group_assgnmt == g)[0]
		anomcnt[g] = np.mean(lkh0[ind])

	maxgroupind = np.argsort(anomcnt)
	flist = list()
	reclist = list()
	preclist = list()
	auclist = list()

	fpres = open('results.txt','a')
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
			step_num[a0] = tp
			step_recall[a0] = rec[-1]
			step_precision[a0] = prec[-1]
			if (rec[-1]+prec[-1]) > 0:
				step_F1[a0] = 2.0*prec[-1]*rec[-1]/(rec[-1]+prec[-1])
			if Ssize < 2:
				step_auc[a0] = 0.0
			else:
				step_auc[a0] = metrics.auc(rec, prec)

		# assign this group to the lbl of argmax(step_num)
		a0 = np.argmax(step_num)
		flist.append(step_F1[a0])
		reclist.append(step_recall[a0])
		preclist.append(step_precision[a0])
		auclist.append(step_auc[a0])
		fpres.write('group %d: size = %d, anomscore = %f, anomlbl = %s, recall = %f, precision = %f, auc = %f, f1 = %f\n' %(g,Ssize,anomcnt[g],anomlist[a0],step_recall[a0],step_precision[a0],step_auc[a0],step_F1[a0]))
		
	F1score[m1] = np.mean(flist)
	print('M = %d, F1-score = %f' %(M, F1score[m1]))
	fpres.write('M = %d, F1-score = %f\n' %(M, F1score[m1]))
	fpres.close()

fpres = open('results.txt','a')
fpres.write('############################################\n')
amax = np.argmax(F1score)
m1 = Mlist[amax]
fpres.write('Best F1-score: M = %d, F1-score = %f' %(m1,F1score[amax]))
fpres.close()
