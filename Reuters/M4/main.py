import os
import numpy as np
from sklearn.cluster import KMeans
import score

LDAPath = '/home/studadmin/Dropbox/ATDFinal/lda'
path = '/home/studadmin/Dropbox/ATDFinal/Reuters'
M4Path = '/home/studadmin/Dropbox/ATDFinal/M4'

anom_per = 50
seed0 = 1819114101
np.random.seed(seed0)

anomlist = ['coffee','ship']
trainfile = path + '/data/trdocs.txt'
trgroupfile = path + '/data/trgroups.txt'
testfile = path + '/data/tdocs'+str(anom_per) + '.txt'
tgroupfile = path + '/data/tgroups'+str(anom_per) + '.txt'
tlblfile = path + '/data/tlbls'+str(anom_per) + '.txt'
resfile = 'results.txt'
fpres = open(resfile,'w')
fpres.write("")
fpres.close()

Mlist = np.arange(12,30,2)
Tlist = np.arange(6,32,4)
for M in Mlist:
	G = M
	Gtest = M
	# run lda on training data
	seed = np.random.randint(seed0)
	cmdtxt = LDAPath + '/lda ' + str(seed) + ' est 0.1 ' + str(M) + ' ' + LDAPath + '/settings.txt ' + trainfile + ' seeded dirlda' 
	os.system(cmdtxt + ' > /dev/null')

	# read topic proportions
	trtheta = np.loadtxt('dirlda/final.gamma')
	sumtheta = np.sum(trtheta,1)
	trtheta = trtheta/sumtheta.reshape(-1,1)

	# clustering
	group_assgnmt = np.argmax(trtheta,1)

	# write group file
	np.savetxt(trgroupfile,group_assgnmt.reshape(-1,1),'%d')


	for T in Tlist:
		#if (T <= 22) and (M == 30):
		#	continue
		#save beta to init K4
		beta = np.loadtxt('dirlda/final.beta')
		os.system('mkdir -p dirM4')
		np.savetxt('dirM4/init.beta', beta.T,'%.10f')
		# run clustering on group centers to find typical sets (genres)
		km2 = KMeans(n_clusters = T)
		km2.fit(trtheta)
		np.savetxt('dirM4/init.alpha', np.log(km2.cluster_centers_), '%.10f')
		pi0 = np.array([np.mean(km2.labels_==j) for j in range(T)])
		np.savetxt('dirM4/init.pi', np.log(pi0).reshape(1,T), '%.10f')

		# Run M4
		seed = np.random.randint(seed0)
		cmdtxt = M4Path + '/M4 '+str(seed) + ' train '+ trainfile + ' ' + trgroupfile + ' ' +str(M)+' '+str(G)+ ' '+str(T)+ ' load '+path+'/M4/dirM4 ' + path + '/M4/dirM4/init' 
		os.system(cmdtxt + ' > /dev/null')

		######################### test set
		# run lda on test set
		seed = np.random.randint(seed0)
		cmdtxt = LDAPath + '/lda ' + str(seed) + ' inf ' + LDAPath + '/settings.txt dirlda/final ' + testfile + ' dirlda/test' 
		os.system(cmdtxt + ' > /dev/null')

		# read test topic proportions
		test_theta = np.loadtxt('dirlda/test-gamma.dat')
		sumtheta = np.sum(test_theta,1)
		test_theta = test_theta/sumtheta.reshape(-1,1)

		# clustering
		group_assgnmt = np.argmax(test_theta,1)

		# write group file
		np.savetxt(tgroupfile,group_assgnmt.reshape(-1,1),'%d')

		# Run M4
		seed = np.random.randint(seed0)
		cmdtxt = M4Path + '/M4 '+str(seed) + ' test '+ testfile + ' ' + tgroupfile + ' '+str(Gtest)+ ' '+path+'/M4/dirM4/final ' + path + '/M4/dirM4' 
		os.system(cmdtxt + ' > /dev/null')

		# compute score
		seed = np.random.randint(seed0)
		f1scores = score.compute_score(seed, path, tlblfile, tgroupfile, resfile, anomlist)
		fpres = open(resfile,'a')
		fpres.write('****** M = %d, T = %d, F1-score = %f\n'%(M,T,np.mean(f1scores)))
		fpres.close()

fpres = open('results.txt','a')
fpres.write('############################################\n')
amax = np.argmax(f1scores)
m1 = amax/len(Tlist)
n1 = amax%len(Tlist)
fpres.write('Best F1-score: M = %d, K = %d, F1-score = %f' %(Mlist[m1],Tlist[n1],f1scores[m1,n1]))
fpres.close()
