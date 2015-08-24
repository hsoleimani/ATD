import os
import numpy as np
from sklearn import metrics

path = '/home/studadmin/Dropbox/ATDFinal/Synthetic'

anom_per = 100

trainfile = path + '/data/trdocs.txt'
testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
tlblfile = path + '/data/tlbls'+str(anom_per) + '.txt'

anomlist = ['11','10']
testlblfile = path + '/data/tlbls'+str(anom_per)+'.txt'
fp = open(testlblfile)
lbllist = fp.readlines()
fp.close()
# count total number of anomalies
Danom = 0.0;
for a0,anomlbl in enumerate(anomlist):
	a = [1 for x in lbllist if anomlbl in x]
	Danom += float(len(a))


## ATD
Ssize = 0
rec = list()
prec = list()
i = 0
tp = 0.0
for step in range(1,3):
	if step == 1:
		testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
		testlblfile = path + '/data/tlbls'+str(anom_per)+'.txt'
	else:
		testfile = path + '/data/tdocs'+str(anom_per)+'_'+str(step)+'.txt'
		testlblfile = path + '/data/tlbls'+str(anom_per)+'_'+str(step)+'.txt'

	Ssize_step = np.loadtxt(path + '/dirtest' + str(step) + '/lkhratio.txt',delimiter = '\t').shape[0] - 1
	Ssize += Ssize_step

	fp = open(path + '/dirtest' + str(step) + '/test.sdocs')

	for l in range(Ssize_step):
		doc = fp.readline()
		if len(doc)==0:
			break
		for anomlbl in anomlist:
			if anomlbl in doc.split()[0]:
				tp += 1.0
				break
		prec.append(tp/(i+1.0))
		rec.append(tp/Danom)
		i += 1

	fp.close()

rec = np.array(rec)
prec = np.array(prec)
if (rec[-1]+prec[-1]) > 0:
	F1 = 2.0*prec[-1]*rec[-1]/(rec[-1]+prec[-1])
auc = metrics.auc(rec, prec)
print('ATD method: auc = %f, F1 = %f' %(auc,F1))



################# baseline method
lkh0 = np.loadtxt(path+'/dirtest1/test.normlkh0')

ind = np.argsort(lkh0)
rec = np.zeros(Ssize)
prec = np.zeros(Ssize)
tp = 0
for n in range(Ssize):
	doclbl = lbllist[ind[n]].split()[0]
	for anomlbl in anomlist:
		if anomlbl in doclbl:
			tp += 1.0
			break
	prec[n] = tp/(n+1.0)
	rec[n] = tp/Danom

if (rec[-1]+prec[-1]) > 0:
	F1 = 2.0*prec[-1]*rec[-1]/(rec[-1]+prec[-1])
auc = metrics.auc(rec, prec)

print('baseline method: auc = %f, F1 = %f' %(auc,F1))

