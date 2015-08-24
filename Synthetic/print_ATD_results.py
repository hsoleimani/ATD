# Prints results of ATD

import os
import numpy as np
from sklearn import metrics

path = '/home/studadmin/Dropbox/ATDFinal/Synthetic'

anom_per = 100
Ssize = -1

trainfile = path + '/data/trdocs.txt'
trainlblfile = path + '/data/trlbls.txt'
validfile = path + '/data/vdocs.txt'
nulltestfile = path + '/data/ntdocs.txt'
nulltestlblfile = path + '/data/ntlbls.txt'
testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
testlblfile = path + '/data/tlbls'+str(anom_per)+'.txt'

#read p-value
pval = np.loadtxt('btstpdir/pvalue.txt')[:,1]
anomlist = ['11','10']
# step >= 1
for step in range(1,4):
	if step == 1:
		testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
		testlblfile = path + '/data/tlbls'+str(anom_per)+'.txt'
	else:
		testfile = path + '/data/tdocs'+str(anom_per)+'_'+str(step)+'.txt'
		testlblfile = path + '/data/tlbls'+str(anom_per)+'_'+str(step)+'.txt'

	Ssize = np.loadtxt('dirtest' + str(step) + '/lkhratio.txt',delimiter = '\t').shape[0] - 1
	
	fp = open(testlblfile)
	lbllist = fp.readlines()
	fp.close()

	step_num = np.zeros(len(anomlist))
	step_recall = np.zeros(len(anomlist))
	step_precision = np.zeros(len(anomlist))
	step_F1 = np.zeros(len(anomlist))
	step_auc = np.zeros(len(anomlist))
	step_base_recall = np.zeros(len(anomlist))
	step_base_precision = np.zeros(len(anomlist))
	step_base_F1 = np.zeros(len(anomlist))
	step_base_auc = np.zeros(len(anomlist))

	for a0,anomlbl in enumerate(anomlist):
		a = [1 for x in lbllist if anomlbl in x]
		Danom = float(len(a))
		if Danom==0:
			step_recall[a0] = 0
			step_precision[a0] = 0
			step_auc[a0] = 0
			step_base_recall[a0] = 0
			step_base_precision[a0] = 0
			step_base_auc[a0] = 0
			continue

		rec = np.zeros(Ssize)
		prec = np.zeros(Ssize)
		fp = open('dirtest' + str(step) + '/test.sdocs')
		
		tp = 0
		for l in range(Ssize):
			doc = fp.readline()
			if len(doc)==0:
				break
			if anomlbl in doc.split()[0]:
				tp += 1.0
			prec[l] = tp/(l+1.0)
			rec[l] = tp/Danom

		fp.close()

		step_num[a0] = tp
		step_recall[a0] = rec[-1]
		step_precision[a0] = prec[-1]
		if (rec[-1]+prec[-1]) > 0:
			step_F1[a0] = 2.0*prec[-1]*rec[-1]/(rec[-1]+prec[-1])
		step_auc[a0] = metrics.auc(rec, prec)
	a0 = np.argmax(step_num)

	print(' **** step = %d, lbl = %s, Ssize = %d, p-value = %f' %(step, anomlist[a0], Ssize, pval[step-1]))
	print('APD: recall = %f, precision = %f, AUC = %f, F1 = %f' %(step_recall[a0], step_precision[a0], step_auc[a0], step_F1[a0]))

