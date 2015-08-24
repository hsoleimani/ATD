# Runs ATD on synthetic data to detect candidate anomalous clsuters

import os
import numpy as np
import remove_docs


ATDPath = '/home/studadmin/Dropbox/ATDFinal/ATDCode'
path = '/home/studadmin/Dropbox/ATDFinal/Synthetic'

anom_per = 100
M = 10
step = 1
seed0 = 1819114101
np.random.seed(seed0)

trainfile = path + '/data/trdocs.txt'
trainlblfile = path + '/data/trlbls.txt'
validfile = path + '/data/vdocs.txt'
nulltestfile = path + '/data/ntdocs.txt'
nulltestlblfile = path + '/data/ntlbls.txt'

# load topics from PTM; 
beta = np.exp(np.loadtxt(path + '/PTM/dir'+str(M)+'/final.beta'))
uswitch = np.loadtxt(path + '/PTM/dir'+str(M)+'/final.u') 
for j in range(0,M):
	ind = np.where(uswitch[:,j]==0)[0]
	beta[ind,j+1] = beta[ind ,0]
N = beta.shape[0]
temp = np.hstack((beta[:,1:M+1],beta[:,0].reshape(N,1)))
os.system('mkdir -p dirnull')
np.savetxt('dirnull/finalPTMshared.beta',np.log(temp+1e-50),'%f')
os.system('cp ' + path + '/PTM/dir'+str(M)+'/final.other '+ 'dirnull/finalPTMshared.other')

for step in range(1,4):
	if step == 1:
		testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
		testlblfile = path + '/data/tlbls'+str(anom_per)+'.txt'
	else:
		testfile = path + '/data/tdocs'+str(anom_per)+'_'+str(step)+'.txt'
		testlblfile = path + '/data/tlbls'+str(anom_per)+'_'+str(step)+'.txt'

	# ATD on test set
	seed = np.random.randint(seed0)
	cmdtxt = ATDPath + '/ATD '+str(seed)+' detec ' + testfile + ' dirtest' + str(step) +' dirnull/finalPTMshared ' + testlblfile + ' 70 ' + validfile + ' 1'
	#print(cmdtxt)
	os.system(cmdtxt)

	Ssize = np.loadtxt('dirtest' + str(step) + '/lkhratio.txt',delimiter = '\t').shape[0] - 1
	print(Ssize)

	remove_docs.remove_sdocs(path, step+1, Ssize, anom_per)


