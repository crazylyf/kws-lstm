def readAndSort():
	file = open('../align.map.mlf', 'r')
	name_end = {}
	content = {}
	vocab = []
	## reading ##
	line = file.readline()
	while (line):
		if (line[0]=='"'):
			curname = line.strip()
			content[curname] = line
			name_end[curname] = 0
		elif (line[0]=='.'):
			content[curname] = content[curname] + line
		else:
			content[curname] = content[curname] + line
			lst = line.split()
			name_end[curname] = int(lst[1])
			if (lst[2] not in vocab):
				vocab.append(lst[2])
		line = file.readline()
	
	file.close()
	
	## sorting and writing ##
	threshold = 1000000
	curfile = 1
	file = open('../mlf/sortedmlf'+('%d' % curfile), 'w')
	curMinLength = 0
	name_end_sorted = sorted(name_end.items(), key = lambda x:x[1])
	for item in name_end_sorted:
		if (curMinLength==0):
			curMinLength = item[1]
		else:
			if (item[1]-curMinLength>threshold):
				curfile = curfile+1
				file.close()
				file = open('../mlf/sortedmlf'+('%d' % curfile), 'w')
				curMinLength = item[1]
		file.write(content[item[0]])
	
	file.close()
	## writing vocabulary ##
	file = open('../vocab/vocab', 'w')
	vocab.sort()
	for i in vocab:
		file.write(i+'\n')
	file.close()

if __name__ == "__main__":
	readAndSort()
