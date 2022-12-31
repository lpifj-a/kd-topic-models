from summarizer import Summarizer
import numpy as np
import pickle
import re
import string
from tqdm import tqdm
from utils import load_jsonlist, save_jsonlist, load_sparse, save_sparse, load_json, save_json

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = "".join(punct_chars)
replace = re.compile("[%s]" % re.escape(punctuation))
alpha = re.compile("^[a-zA-Z_]+$")
alpha_or_num = re.compile("^[a-zA-Z_]+|[0-9_]+$")
alphanum = re.compile("^[a-zA-Z0-9_]+$")


def clean_text(
    text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False
):
    # remove html tags
    if strip_html:
        text = re.sub(r"<[^>]+>", "", text)
    else:
        # replace angle brackets
        text = re.sub(r"<", "(", text)
        text = re.sub(r">", ")", text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r"\S+@\S+", " ", text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r"\s@\S+", " ", text)
    # replace underscores with spaces
    text = re.sub(r"_", " ", text)
    # break off single quotes at the ends of words
    text = re.sub(r"\s\'", " ", text)
    text = re.sub(r"\'\s", " ", text)
    # remove periods
    # text = re.sub(r"\.", "", text)
    # replace all other punctuation (except single quotes) with spaces
    # text = replace.sub(" ", text)
    # remove single quotes
    text = re.sub(r"\'", "", text)
    # replace all whitespace with a single space
    text = re.sub(r"\s", " ", text)
    # strip off spaces on either end
    text = text.strip()
    return text


model = Summarizer()

data = load_jsonlist("/home/watanabe/kd-topic-models/data/20ng/aligned/train.jsonlist")

teacher_embs = np.empty((len(data),1024))

for i,d in enumerate(tqdm(data)):
    text = d["text"]
    text = clean_text(text, strip_html=True, lower=True, keep_emails=False, keep_at_mentions=False)
    # emb = model.run_embeddings(text,num_sentences=3,aggregate='mean')
    emb = model.run_embeddings(text,num_sentences=3,aggregate='max')
    teacher_embs[i] = emb
    isnan = np.isnan(teacher_embs[i])
    if any(isnan):
        print("NaN is found")
        print("index:",i)
        print("text:",text)
        print("embeddings:",emb)
    
np.save("teacher_emb/20ng/train_teacher_emb_max_pooling",teacher_embs)

