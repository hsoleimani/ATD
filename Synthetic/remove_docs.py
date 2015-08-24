# This code is used to remove documents in a detected cluster 
# from the test batch. This is used when there are more than 
# one anomalous topic in the test set and we detect those topics
# one by one.

import numpy as np
def remove_sdocs(path, step, Ssize, anom_per):
	if step == 2:
		testfile = path + '/data/tdocs'+str(anom_per)+'.txt'
		testtitlefile = path + '/data/tlbls'+str(anom_per)+'.txt'
		ntestfile =  path + '/data/ntdocs.txt'
	else:
		testfile = path + '/data/tdocs'+str(anom_per)+'_'+str(step-1)+'.txt'
		testtitlefile = path + '/data/tlbls'+str(anom_per)+'_'+str(step-1)+'.txt'
		ntestfile =  path + '/data/ntdocs_'+str(step-1)+'.txt'

	testfile2 = path + '/data/tdocs'+str(anom_per)+'_'+str(step)+'.txt'
	testtitlefile2 = path + '/data/tlbls'+str(anom_per)+'_'+str(step)+'.txt'

	# need to also prepare ntest file
	ntestfile2 =  path + '/data/ntdocs_'+str(step)+'.txt'


	#read in test docs
	fp1 = open(testfile,'r')
	testdocs = fp1.readlines()
	fp1.close()
	fp2 = open(testtitlefile,'r')
	testtitles = fp2.readlines()
	fp2.close()

	ntest = len(testdocs)

	# list of docs in S
	fp1 = open('dirtest' + str(step-1) + '/test.sdocs')
	docind = np.zeros(ntest,dtype=np.int)
	Sdocstxt = list()
	d = 0
	while True:
		dl = fp1.readline()
		if len(dl)==0:
			break
		docind[d] = int(dl.split()[1])
		Sdocstxt.append(testdocs[docind[d]])
		d += 1
		if d == Ssize:
			break
	Ssize = d
	fp1.close()

	fp1 = open(testfile2, 'w+')
	fp2 = open(testtitlefile2, 'w+')
	for d in range(ntest):
		if d in docind:
			continue
		fp1.write(testdocs[d])
		fp2.write(testtitles[d])
	fp1.close()
	fp2.close()

	# prepare null test file (used for btstp)
	fp1 = open(ntestfile, 'r')
	fp2 = open(ntestfile2, 'w+')
	while True:
		doc = fp1.readline()
		if len(doc) == 0:
			break
		if doc in Sdocstxt:
			continue
		fp2.write(doc)
	fp1.close()
	fp2.close()


