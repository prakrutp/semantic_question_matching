import csv
import re

#data_path = '../data/sample_data.tsv'
data_path = '../../Data/quora_duplicate_questions.tsv'

def process_question(q):
	clean_q = ''
	q = q.lower()
	q = re.sub(r'[^a-zA-Z0-9 ]','', q)
	words = q.split()
	for word in words:
		# Add space when transitioning from num to alpha
		match = re.match(r"([a-z]+)([0-9]+)", word, re.I)
		if match:
			items = match.groups()
			for item in items:
				clean_q += item + ' '
		else:
			clean_q += word + ' '
	return clean_q.strip()


def get_clean_data():
	with open(data_path) as csvfile:
		reader = csv.reader(csvfile, delimiter='\t')
		next(reader) # skip header
		clean_data = []
		for row in reader:
			pair_id = int(row[0])
			q1_id = int(row[1])
			q2_id = int(row[2])
			q1 = process_question(row[3])
			q2 = process_question(row[4])
			duplicate = row[5]
			clean_data.append(q1 + '\t' + q2 + '\t' + duplicate)
	return clean_data

clean_data = get_clean_data()



