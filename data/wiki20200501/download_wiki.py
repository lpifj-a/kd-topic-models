from datasets import load_dataset
import json
import file_handling as fh

train_file = 'train.jsonlist'
doc_list = []

corpus = load_dataset('wikipedia', '20200501.en', split = 'train')

for l_i, line in enumerate(corpus['text']):
    # Display occassional progress
    if (l_i +1) % 1000000 == 0:
        print("Processed {:d} / 6078422".format(l_i+1))
    doc = {'text': line}
    doc_list.append(doc)

print("Saving processed data")
fh.write_jsonlist(doc_list, train_file)
