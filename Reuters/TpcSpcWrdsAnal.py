fp = open('/home/studadmin/Dropbox/ATDFinal/Reuters/data/vocabs.txt')
words = fp.readlines()
fp.close()

step = 1
fpname = 'dirtest'+str(step)

onwrds = np.loadtxt(fpname+'/test.beta')[:,1]
inwrds = np.loadtxt(fpname+'/test.beta')[:,2]
beta = np.loadtxt(fpname+'/test.beta')[:,0]

highprobs = np.argsort(-beta)
'''wrdstr = []
for i in range(20):
	wrdstr.append(words[highprobs[i]].split()[0])
print(', '.join(wrdstr))'''


# print high prob occurring topic-specific words:
cnt = 0
wrdstr = []
for i in highprobs:
	if (onwrds[i] == 1) and (inwrds[i] == 1):
		cnt += 1
		wrdstr.append(words[i].split()[0])
	if cnt == 20:
		break
print(', '.join(wrdstr))


# compute rank of occurring topic-specific words
wrdrnk = highprobs[np.where(onwrds*inwrds == 1)[0]]
#print(np.median(wrdrnk))
#print(np.mean(wrdrnk))

# compute rank of non-occurring topic-specific words
ind = np.where((onwrds==1)*(inwrds==0)==1)[0]
wrdrnk = highprobs[ind]
#print(np.median(wrdrnk))
#print(np.mean(wrdrnk))

# print high prob occurring shared words:
cnt = 0
wrdstr = []
for i in highprobs:
	if (onwrds[i] == 0) and (inwrds[i] == 1):
		cnt += 1
		wrdstr.append(words[i].split()[0])
	if cnt == 20:
		break
print(', '.join(wrdstr))

# print *low* prob occurring topic-specific words:
cnt = 0
wrdstr = []
for i in np.argsort(beta):
	if (onwrds[i] == 1) and (inwrds[i] == 1):
		cnt += 1
		wrdstr.append(words[i].split()[0])
	if cnt == 20:
		break
print(', '.join(wrdstr))


# print *low* prob non-occurring topic-specific words:
cnt = 0
wrdstr = []
for i in np.argsort(beta):
	if (onwrds[i] == 1) and (inwrds[i] == 0):
		cnt += 1
		wrdstr.append(words[i].split()[0])
	if cnt == 20:
		break
print(', '.join(wrdstr))

