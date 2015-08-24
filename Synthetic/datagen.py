# Creates synthetic data

import numpy as np
import random as random
import os

seed = 1415117801

path = 'data'
os.system('mkdir -p '+path)
np.random.seed(seed)
M = 10
N = 3000
D = M*300
lam = 50

beta = np.zeros((N,M))
n1 = int(np.floor(N*0.01))
n2 = int(np.floor(N*0.1))
remainingwrds = np.arange(N)
highprobvec = np.zeros((n1,M+1))
for j in range(M):
	highprobs = np.random.choice(np.arange(int(np.floor(N/M))), n1, replace = False)
	highprobs = highprobs + j*int(np.floor(N/M))
	highprobvec[:,j] = highprobs
	#beta[highprobs,j] = 0.7 + 0.1*np.random.uniform(0, 1, n1)
	beta[highprobs,j] = 10. + np.random.uniform(1, 2, n1)
	beta[np.setdiff1d(np.arange(N),highprobs),j] = .1*np.random.uniform(0, 1, N-n1)
	#beta[:,j] = beta[:,j]/np.sum(beta[:,j])
	beta[:,j] = np.random.dirichlet(beta[:,j])
	remainingwrds = np.setdiff1d(remainingwrds, highprobs)


np.savetxt(path+'/beta.txt', beta, '%5.10f')

#generate trainig docs
while True:
	fp1 = open(path+'/trdocs.txt','w')
	fp2 = open(path+'/trlbls.txt','w')
	#fp3 = open(path+'/traintheta.txt','w')
	wchk = np.zeros(N)
	for d in range(D):
		j = d%M
		theta = np.ones(M)
		theta[j] = 50.0
		theta = theta/np.sum(theta)

		ld = 0
		nd = np.random.poisson(lam, 1)[0]
		doc = {}
		for i in range(nd):
			z = np.random.choice(M, 1, p = theta)[0]
			w = np.random.choice(N, 1, p = beta[:,z])[0]
			if w in doc:
				doc[w] += 1
			else:
				doc.update({w:1})
				ld += 1
				wchk[w] = 1
		fp1.write(str(ld) + ' ')
		for w in doc:
			fp1.write(str(w)+':'+str(doc[w])+' ')
		fp1.write('\n')
		fp2.write(str(j) + '\n')

	fp1.close()
	fp2.close()
	break
	if (np.sum(wchk==0) == 0):
		break


#generate null test docs
fp1 = open(path+'/ntdocs.txt','w')
fp2 = open(path+'/ntlbls.txt','w')
Dtest = int(np.floor(M*300))
for d in range(Dtest):
	j = d%M
	theta = np.ones(M)
	theta[j] = 50.0
	theta = theta/np.sum(theta)
	ld = 0
	nd = np.random.poisson(lam, 1)[0]
	doc = {}
	for i in range(nd):
		z = np.random.choice(M, 1, p = theta)[0]
		w = np.random.choice(N, 1, p = beta[:,z])[0]
		if wchk[w] == 0:
			continue
		if w in doc:
			doc[w] += 1
		else:
			doc.update({w:1})
			ld += 1
	fp1.write(str(ld) + ' ')
	for w in doc:
		fp1.write(str(w)+':'+str(doc[w])+' ')
	fp1.write('\n')
	fp2.write(str(j) + '\n')

fp1.close()
fp2.close()


#generate validation docs
fp1 = open(path+'/vdocs.txt','w')
fp2 = open(path+'/vlbls.txt','w')
Dtest = int(np.floor(M*300))
for d in range(Dtest):
	j = d%M
	theta = np.ones(M)
	theta[j] = 50.0
	theta = theta/np.sum(theta)
	ld = 0
	nd = np.random.poisson(lam, 1)[0]
	doc = {}
	for i in range(nd):
		z = np.random.choice(M, 1, p = theta)[0]
		w = np.random.choice(N, 1, p = beta[:,z])[0]
		if wchk[w] == 0:
			continue
		if w in doc:
			doc[w] += 1
		else:
			doc.update({w:1})
			ld += 1
	fp1.write(str(ld) + ' ')
	for w in doc:
		fp1.write(str(w)+':'+str(doc[w])+' ')
	fp1.write('\n')
	fp2.write(str(j) + '\n')

fp1.close()
fp2.close()

# generate one additional topic
M = M + 1
beta = np.hstack((beta, np.zeros((N,1))))
n1 = int(np.floor(N*0.01))
j = M-1
highprobs = np.random.choice(np.arange(int(np.floor(N/M))), n1, replace = False)
highprobs = highprobs + j*int(np.floor(N/M))
highprobvec[:,j] = highprobs
beta[highprobs,j] = 10. + np.random.uniform(1, 2, n1)
beta[np.setdiff1d(np.arange(N),highprobs),j] = .1*np.random.uniform(0, 1, N-n1)
beta[:,j] = np.random.dirichlet(beta[:,j])

# generate the second anomalous topic  
M = M + 1
beta = np.hstack((beta, np.zeros((N,1))))
n1 = int(np.floor(N*0.01))
j = M-1
highprobvec = np.hstack((highprobvec, np.zeros((n1,1))))
highprobs = np.random.choice(np.arange(int(np.floor(N/M))), n1, replace = False)
highprobs = highprobs + j*int(np.floor(N/M))
highprobvec[:,j] = highprobs
beta[highprobs,j] = 10. + np.random.uniform(1, 2, n1)
beta[np.setdiff1d(np.arange(N),highprobs),j] = .1*np.random.uniform(0, 1, N-n1)
beta[:,j] = np.random.dirichlet(beta[:,j])

np.savetxt(path+'/beta.txt', beta, '%5.10f')
np.savetxt(path+'/highprobs.txt', highprobvec, '%d')

#generate test docs
fp1 = open(path+'/tdocs100.txt','w')
fp2 = open(path+'/tlbls100.txt','w')
Dtest = int(np.floor(M*100))
for j in range(M):
	if j < 10:
		DD = 200
	else:
 		DD = 30
	for d in range(DD):
		theta = np.ones(M)
		theta[j] = 50.0
		if j == M-1: #anomalous topic
			jj = 8
			theta[jj] = 50.0 #half of each documnet is anomalous
		if j == M-2:
			jj = 0
			theta[jj] = 50.0 #half of each documnet is anomalous
		theta = theta/np.sum(theta)
		ld = 0
		nd = np.random.poisson(lam, 1)[0]
		doc = {}
		for i in range(nd):
			z = np.random.choice(M, 1, p = theta)[0]
			w = np.random.choice(N, 1, p = beta[:,z])[0]
			if wchk[w] == 0:
				continue
			if w in doc:
				doc[w] += 1
			else:
				doc.update({w:1})
				ld += 1
		fp1.write(str(ld) + ' ')
		for w in doc:
			fp1.write(str(w)+':'+str(doc[w])+' ')
		fp1.write('\n')
		fp2.write(str(j) + '\n')

fp1.close()
fp2.close()




