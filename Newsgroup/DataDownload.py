import re, os
import urllib2

# download and extract http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
request = urllib2.urlopen('http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz')
output = open("20news-bydate.tar.gz","w")
output.write(request.read())
output.close()
os.system('mkdir -p 20news-bydate')
os.system('tar -zxvf 20news-bydate.tar.gz -C 20news-bydate')


# crawl all folders
# remove lines that start with "From:", "Reply-To:", or "Lines:"
# remove 

fptrain = open('rawtrain.txt','w')
# read training docs
directory = '20news-bydate/20news-bydate-train'
d = 0
for path, dirs, files in os.walk(directory):
	for f in files:
		filename = os.path.join(path, f)
		
		lbl = path.split('/')[-1] # class label
		
		# read file
		fp = open(filename)
		docraw = ''
		while True:
			ln = fp.readline()
			if len(ln)==0:
				break
			# remove this line if it starts with:
			if re.match('From: ', ln) != None:
				continue
			if re.match('Reply-To: ', ln) != None:
				continue
			if re.match('Lines: ', ln) != None:
				continue
			if re.match('^Organization: ', ln) != None:
				continue
			if re.match('Subject: ', ln) != None: # if it starts with 'Subject:', remove it but keep the rest of the line
				ln = re.sub(r'^Subject: ', ' ', ln)
			ln = ln.lower()
			ln = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}', ' ', ln) # remove email addresses
			ln = re.sub(r'in article .* writes', ' ', ln) # remove 'in article <name> writes'
			ln = re.sub(r'[^a-zA-Z]', ' ', ln)	# remove non-letters
			ln = re.sub(r' {2,}', ' ', ln)	# sub 2 or more consecutive spaces with only 1
			lntxt = [x for x in ln.split() if len(x)>=3] # remove words with <= 2 characters
			docraw += " ".join(lntxt) + ' '

		fp.close()
		docraw = re.sub(r' {2,}', ' ', docraw)	# sub 2 or more consecutive spaces with only 1
		
		fptrain.write(lbl + ' ' + docraw + '\n')
		d += 1

		#if d%5000 == 0:
		#	print(d)
fptrain.close()

#print('Done with training docs')

		

fptest = open('rawtest.txt','w')
# read training docs
directory = '20news-bydate/20news-bydate-test'
d = 0
for path, dirs, files in os.walk(directory):
	for f in files:
		filename = os.path.join(path, f)
		
		lbl = path.split('/')[-1] # class label
		
		# read file
		fp = open(filename)
		docraw = ''
		while True:
			ln = fp.readline()
			if len(ln)==0:
				break
			# remove this line if it starts with:
			if re.match('From: ', ln) != None:
				continue
			if re.match('Reply-To: ', ln) != None:
				continue
			if re.match('Lines: ', ln) != None:
				continue
			if re.match('^Organization: ', ln) != None:
				continue
			if re.match('Subject: ', ln) != None: # if it starts with 'Subject:', remove it but keep the rest of the line
				ln = re.sub(r'^Subject: ', ' ', ln)
			ln = ln.lower()
			ln = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}', ' ', ln) # remove email addresses
			ln = re.sub(r'in article .* writes', ' ', ln) # remove 'in article <name> writes'
			ln = re.sub(r'[^a-zA-Z]', ' ', ln)	# remove non-letters
			ln = re.sub(r' {2,}', ' ', ln)	# sub 2 or more consecutive spaces with only 1
			lntxt = [x for x in ln.split() if len(x)>=3] # remove words with <= 2 characters
			docraw += " ".join(lntxt) + ' '

		fp.close()
		docraw = re.sub(r' {2,}', ' ', docraw)	# sub 2 or more consecutive spaces with only 1
		
		fptest.write(lbl + ' ' + docraw + '\n')
		d += 1

		#if d%5000 == 0:
		#	print(d)
fptest.close()
# delete downloaded/extracted files
os.system('rm -rf 20news-bydate')
os.system('rm 20news-bydate.tar.gz')
