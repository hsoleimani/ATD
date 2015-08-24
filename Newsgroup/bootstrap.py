import numpy as np
import random as random
import os, re


ATDPath = '/home/studadmin/Dropbox/ATDFinal/ATDCode'
path = '/home/studadmin/Dropbox/ATDFinal/Newsgroup'
T = 1000

anom_per = 20
step = 1

seed0 = 1492489482
np.random.seed(seed0)

os.system('mkdir -p btstpdir')
fp_pval = open('btstpdir/pvalue.txt','w')
fp_pval.close()

for step in range(1,5):
	Ssize = np.loadtxt('dirtest' + str(step) + '/lkhratio.txt',delimiter = '\t').shape[0] - 1
	print('step = %d, Ssize = %d' %(step, Ssize))

	if step == 1:
		testfile = path + '/data/tdocs'+str(anom_per)+'.txt'
		testtitlefile = path + '/data/tlbls'+str(anom_per)+'.txt'
		ntestfile = path + '/data/ntdocs.txt'
	else:
		testfile = path + '/data/tdocs'+str(anom_per)+'_'+str(step)+'.txt'
		testtitlefile = path + '/data/tlbls'+str(anom_per)+'_'+str(step)+'.txt'
		ntestfile = path + '/data/ntdocs_'+str(step)+'.txt'

	trainfile = path + '/data/trdocs.txt'
	validfile = path + '/data/vdocs.txt'

	BSvalidfile = path +'/btstpdir/vdocs.txt'
	BSthetafile = path + '/btstpdir/theta0.txt'
	mainscorefile = path + '/dirtest'+str(step)+'/lkhratio.txt'
	mainthetafile = path + '/dirtest'+str(step)+'/test.theta0'
	mainsdocfile = path + '/dirtest'+str(step)+'/test.sdocs'
	doclenfile =  path + '/dirtest'+str(step)+'/doclen.txt'

	os.system('mkdir -p btstpdir')

	# beta
	os.system('cp '+path +'/dirnull/finalPTMshared.beta btstpdir/null.beta')
	os.system('cp '+path +'/dirnull/finalPTMshared.other btstpdir/null.other')

	theta0 = np.loadtxt(mainthetafile, delimiter = ',')

	# read test docs to determine their length
	fp1 = open(testfile,'r')
	testdocs = fp1.readlines()
	fp1.close()

	# read docs in S - write length of docs
	fp1 = open(mainsdocfile)
	fp2 = open(doclenfile,'w')
	Sdocslist = list()
	Docindlist = list()
	d = 0
	while True:
		dl = fp1.readline()
		if len(dl)==0:
			break
		docind = int(dl.split()[1])
		doc = testdocs[docind]
		wrds = re.findall('([0-9]*):[0-9]*',doc)
		cnts = re.findall('[0-9]*:([0-9]*)',doc)
		total = np.sum([np.int(x) for x in cnts])
		if (Ssize == d):
			break
		Sdocslist.append(doc)
		Docindlist.append(docind)
		fp2.write(str(total)+'\n')
		d += 1
	Ssize = d # for when Ssize = -1
	fp1.close()
	fp2.close()

	#prepare theta0file
	np.savetxt(BSthetafile, theta0[Docindlist,:],'%f')

	# prepare validation file
	os.system('cp '+ validfile + ' ' + BSvalidfile)
	# add docs from ntestfile to vdocs
	fpv = open(BSvalidfile,'a')
	fpt = open(ntestfile,'r')
	while True:
		doc = fpt.readline()
		if len(doc)==0:
			break
		if doc in Sdocslist:
			#print('aha')
			continue
		fpv.write(doc)

	fpv.close()
	fpt.close()


	# Run btstp
	seed = np.random.randint(seed0)
	cmdtxt = ATDPath + '/ATD ' + str(seed) +' btstp2 ' + BSvalidfile + ' btstpdir btstpdir/null ' + BSthetafile + ' ' + str(Ssize)
	cmdtxt += ' ' + str(T) + ' ' + doclenfile
	os.system(cmdtxt)# + ' > /dev/null')

	mainscore = np.loadtxt(mainscorefile, delimiter = '\t')[Ssize-1,1]
	btss = np.loadtxt('btstpdir/BSlkhratio.txt', delimiter = '\t')[:,0]
	pvalue = (np.sum(btss > mainscore)+1.0)/(T+1.0)
	print('step = %d, pvalue = %f' %(Ssize,pvalue))
	fp_pval = open('btstpdir/pvalue.txt','a')
	fp_pval.write(str(Ssize) + ' ' + str(pvalue) + ' ' + str(np.percentile(btss, 99)) + '\n')
	fp_pval.close()
