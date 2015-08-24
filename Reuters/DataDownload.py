import re, os
import urllib2

# download and extract http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz
request = urllib2.urlopen('http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz')
output = open("reuters21578.tar.gz","w")
output.write(request.read())
output.close()
os.system('mkdir -p reuters21578')
os.system('tar -zxvf reuters21578.tar.gz -C reuters21578')
os.system('cp reuters21578/all-topics-strings.lc.txt all-topics-strings.txt')


directory = 'reuters21578'

fptrain = open('rawdocs.txt','w')
d = 0
for path, dirs, files in os.walk(directory):
	for f in files:
		filename = os.path.join(path, f)
		if 'reut2' not in filename:
			continue
		#print(filename)
		lbl = path.split('/')[-1] # class label
		
		# read file
		fp = open(filename)
		fileraw = fp.read()
		fp.close()
		docstxt = fileraw.split('</REUTERS>')
		for ln in docstxt:
			if '<TOPICS>' not in ln:
				continue
			if '<TEXT TYPE="BRIEF">' in ln: # breif docs don't have text body
				continue
			# extract topic(s)
			topicstring = ln.split('<TOPICS>')[1].split('</TOPICS>')[0]
			topics = re.findall('<D>([a-zA-Z-]*)</D>',topicstring)
			if len(topics)==0: # skip documents with no labels
				continue
			topictxt = '_'.join(topics) # join topics with '_' if there are more than one
			
			# extract body and title
			if '<BODY>' not in ln:
				continue
			maintxt = ln.split('<BODY>')[1].split('</BODY>')[0] 
			if '<TITLE>' in ln:
				maintxt += ' ' + ln.split('<TITLE>')[1].split('</TITLE>')[0]
			
			maintxt = re.sub(r'\n', ' ', maintxt)	# replace \n with space
			maintxt = maintxt.lower()
			maintxt = re.sub(r'[^a-zA-Z]', ' ', maintxt)	# remove non-letters
			maintxt = re.sub(r'reuter', ' ', maintxt)	# remove 'reuter'
			maintxt = re.sub(r' {2,}', ' ', maintxt)	# sub 2 or more consecutive spaces with only 1
			lntxt = [x for x in maintxt.split() if len(x)>=3] # remove words with <= 2 characters
			docraw = " ".join(lntxt) + ' '
			docraw = re.sub(r' {2,}', ' ', docraw)	# sub 2 or more consecutive spaces with only 1

			fptrain.write(topictxt + ' ' + docraw + '\n')

			d += 1

		#if d%5000 == 0:
		#	print(d)
fptrain.close()

print('# documents: %d' %d)
# delete downloaded/extracted files
os.system('rm -rf reuters21578')
os.system('rm reuters21578.tar.gz')
