import os
import scispacy
import spacy
import string
import re

nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("sentencizer")

input_folder = "input_folder"
output_folder = "output_folder"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

i=0
for filename in os.listdir(input_folder):
    print(i)
    i+=1
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    with open(input_path, "r") as input_file:
        text = input_file.read()
        text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)
        text = re.sub(r'\.\n\s(\w)', r'. \1', text)
        doc = nlp(text)
        with open(output_path, "w") as output_file:
            prev_sent_text = ""
            for sent in doc.sents:
                sent_text = sent.text.replace(" \n", " ").replace("\t", " ").replace(".\n", " ").replace("-\n", "").replace(",\n", ", ")
                sent_text = "".join([char for char in sent_text if char in string.printable])
                if len(sent_text) < 75:
                    continue
                if prev_sent_text:
                    output_file.write(prev_sent_text + " " + sent_text + "\n")
                    prev_sent_text = ""
                else:
                    prev_sent_text = sent_text
