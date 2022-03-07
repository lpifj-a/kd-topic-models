import os
import re
import sys
import string
import json
import itertools
import argparse
import multiprocessing
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat
from gensim.utils import chunkize

import file_handling as fh

"""
Convert a dataset into the required format (as well as formats required by other tools).
Input format is one line per item.
Each line should be a json object.
At a minimum, each json object should have a "text" field, with the document text.
Any other field can be used as a label (specified with the --label option).
If training and test data are to be processed separately, the same input directory should be used
Run "python preprocess_data -h" for more options.
If an 'id' field is provided, this will be used as an identifier in the dataframes, otherwise index will be used 
"""

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = "".join(punct_chars)
replace = re.compile("[%s]" % re.escape(punctuation))
alpha = re.compile("^[a-zA-Z_]+$")
alpha_or_num = re.compile("^[a-zA-Z_]+|[0-9_]+$")
alphanum = re.compile("^[a-zA-Z0-9_]+$")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("train_infile", type=str, help="input file")
    parser.add_argument("output_dir", type=str, help="output directory")

    label_parser = parser.add_mutually_exclusive_group()
    label_parser.add_argument(
        "--label",
        dest="label",
        default=None,
        help="field(s) to use as label (comma-separated)",
    )
    label_parser.add_argument(
        # c.f. https://stackoverflow.com/a/18609361/5712749
        "--label_dicts",
        dest="label_dicts",
        type=json.loads,
        default=None,
        help="""
        field(s) to use as label along with their values, format as json dict, e.g.,
        "{'label_1': ['class_1_a', 'class_1_b'], 'label_2': ['class_2_a', 'class_2_b']}"
        """,
    )
    parser.add_argument(
        "--test", dest="test", default=None, help="Test data (test.jsonlist)",
    )
    parser.add_argument(
        "--train-prefix",
        dest="train_prefix",
        default="train",
        help="Output prefix for training data",
    )
    parser.add_argument(
        "--test-prefix",
        dest="test_prefix",
        default="test",
        help="Output prefix for test data",
    )
    parser.add_argument(
        "--stopwords",
        dest="stopwords",
        default="snowball",
        help="List of stopwords to exclude [None|mallet|snowball]",
    )
    parser.add_argument(
        "--min-doc-count",
        dest="min_doc_count",
        default=0,
        help="Exclude words that occur in less than this number of documents",
    )
    parser.add_argument(
        "--max-doc-freq",
        dest="max_doc_freq",
        default=1.0,
        help="Exclude words that occur in more than this proportion of documents",
    )
    parser.add_argument(
        "--keep-num",
        action="store_true",
        dest="keep_num",
        default=False,
        help="Keep tokens made of only numbers",
    )
    parser.add_argument(
        "--keep-alphanum",
        action="store_true",
        dest="keep_alphanum",
        default=False,
        help="Keep tokens made of a mixture of letters and numbers",
    )
    parser.add_argument(
        "--strip-html",
        action="store_true",
        dest="strip_html",
        default=False,
        help="Strip HTML tags",
    )
    parser.add_argument(
        "--no-lower",
        action="store_true",
        dest="no_lower",
        default=False,
        help="Do not lowercase text",
    )
    parser.add_argument(
        "--min-length", dest="min_length", default=3, help="Minimum token length",
    )
    parser.add_argument(
        "--ngram_min", dest="ngram_min", default=1, help="n-grams lower bound",
    )
    parser.add_argument(
        "--ngram_max", dest="ngram_max", default=1, help="n-grams upper bound",
    )
    parser.add_argument(
        "--vocab-size",
        dest="vocab_size",
        default=None,
        help="Size of the vocabulary (by most common, following above exclusions)",
    )
    parser.add_argument(
        "--workers", default=1, type=int, help="Processes to use when processing",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        default=42,
        help="Random integer seed (only relevant for choosing test set)",
    )

    options = parser.parse_args()

    train_infile = options.train_infile
    output_dir = options.output_dir

    test_infile = options.test
    train_prefix = options.train_prefix
    test_prefix = options.test_prefix
    label_fields = options.label or options.label_dicts
    min_doc_count = int(options.min_doc_count)
    ngram_range = int(options.ngram_min), int(options.ngram_max)
    min_doc_count = int(options.min_doc_count)
    max_doc_freq = float(options.max_doc_freq)
    workers = options.workers

    vocab_size = options.vocab_size
    stopwords = options.stopwords
    if stopwords == "None":
        stopwords = None
    keep_num = options.keep_num
    keep_alphanum = options.keep_alphanum
    strip_html = options.strip_html
    lower = not options.no_lower
    min_length = int(options.min_length)
    seed = options.seed
    if seed is not None:
        np.random.seed(int(seed))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess_data(
        train_infile,
        test_infile,
        output_dir,
        train_prefix,
        test_prefix,
        min_doc_count,
        max_doc_freq,
        ngram_range,
        vocab_size,
        stopwords,
        keep_num,
        keep_alphanum,
        strip_html,
        lower,
        min_length,
        label_fields=label_fields,
        workers=workers,
    )


def preprocess_data(
    train_infile,
    test_infile,
    output_dir,
    train_prefix,
    test_prefix,
    min_doc_count=0,
    max_doc_freq=1.0,
    ngram_range=(1, 1),
    vocab_size=None,
    stopwords=None,
    keep_num=False,
    keep_alphanum=False,
    strip_html=False,
    lower=True,
    min_word_length=3,
    max_doc_length=5000,
    label_fields=None,
    workers=4,
    proc_multiplier=500,
):

    if stopwords == "mallet":
        print("Using Mallet stopwords")
        stopword_list = fh.read_text(os.path.join("stopwords", "mallet_stopwords.txt"))
    elif stopwords == "snowball":
        print("Using snowball stopwords")
        stopword_list = fh.read_text(
            os.path.join("stopwords", "snowball_stopwords.txt")
        )
    elif stopwords is not None:
        print("Using custom stopwords")
        stopword_list = fh.read_text(
            os.path.join("stopwords", stopwords + "_stopwords.txt")
        )
    else:
        stopword_list = []
    stopword_set = {s.strip() for s in stopword_list}

    print("Reading data files")
    train_items = fh.LazyJsonlistReader(train_infile)
    n_train = len(train_items)
    print("Found {:d} training documents".format(n_train))

    if test_infile is not None:
        test_items = fh.LazyJsonlistReader(test_infile)
        n_test = len(test_items)
        print("Found {:d} test documents".format(n_test))
    else:
        test_items = []
        n_test = 0

    n_items = n_train + n_test

    if label_fields:
        label_lists = {}
        if "," in label_fields:
            label_fields = label_fields.split(",")
        else:
            label_fields = [label_fields]
    if label_fields is None:
        label_fields = []
        # labelがないデータを扱うときにlabel_listsが定義されていないというエラーが出たので書き加えた
        print("label_fiels is None")
        label_lists = {}

    # make vocabulary
    train_ids, train_parsed, train_labels = [], [], []
    test_ids, test_parsed, test_labels = [], [], []

    print("Parsing documents")
    word_counts = Counter()
    doc_counts = Counter()

    vocab = None

    # process in blocks
    pool = multiprocessing.Pool(workers)
    chunksize = proc_multiplier * workers

    kwargs = {
        "strip_html": strip_html,
        "lower": lower,
        "keep_numbers": keep_num,
        "keep_alphanum": keep_alphanum,
        "min_length": min_word_length,
        "stopwords": stopword_set,
        "ngram_range": ngram_range,
        "vocab": vocab,
        "label_fields": label_fields,
    }

    # these two loops below do the majority of the preprocessing. unfortunately, without
    # a major refactor, they cannot be turned into generators and the results of
    # tokenization must be appended to a list. this unfortunately implies a large
    # memory footprint
    for i, group in enumerate(chunkize(iter(train_items), chunksize=chunksize)):
        print(f"On training chunk {i} of {len(train_items) // chunksize}", end="\r")
        for ids, tokens, labels in pool.imap(partial(_process_item, **kwargs), group):
            # store the parsed documents
            if ids is not None:
                train_ids.append(ids)
            if labels is not None:
                train_labels.append(labels)
            tokens = tokens[:max_doc_length]

            # keep track of the number of documents with each word
            word_counts.update(tokens)
            doc_counts.update(set(tokens))
            train_parsed.append(" ".join(tokens))  # more efficient storage

    print("Train set processing complete")

    for i, group in enumerate(chunkize(iter(test_items), chunksize=chunksize)):
        print(f"On testing chunk {i} of {len(test_items) // chunksize}", end="\r")
        for ids, tokens, labels in pool.imap(partial(_process_item, **kwargs), group):
            # store the parsed documents
            if ids is not None:
                test_ids.append(ids)
            if labels is not None:
                test_labels.append(labels)
            tokens = tokens[:max_doc_length]

            # keep track of the number of documents with each word
            word_counts.update(tokens)
            doc_counts.update(set(tokens))
            test_parsed.append(" ".join(tokens))  # more efficient storage

    print("Test set processing complete")
    pool.terminate()

    print("Size of full vocabulary=%d" % len(word_counts))

    # store possible label values
    if label_fields:
        labels_df = pd.DataFrame.from_records(train_labels + test_labels)
    for label_name in label_fields:
        label_list = sorted(labels_df[label_name].unique().tolist())
        n_labels = len(label_list)
        print("Found label %s with %d classes" % (label_name, n_labels))
        label_lists[label_name] = label_list

    print("Selecting the vocabulary")
    most_common = doc_counts.most_common()
    words, doc_counts = zip(*most_common)
    doc_freqs = np.array(doc_counts) / float(n_items)
    vocab = [
        word
        for i, word in enumerate(words)
        if doc_counts[i] >= min_doc_count and doc_freqs[i] <= max_doc_freq
    ]
    most_common = [word for i, word in enumerate(words) if doc_freqs[i] > max_doc_freq]
    if max_doc_freq < 1.0:
        print(
            "Excluding words with frequency > {:0.2f}:".format(max_doc_freq),
            most_common,
        )

    print("Vocab size after filtering = %d" % len(vocab))
    if vocab_size is not None:
        if len(vocab) > int(vocab_size):
            vocab = vocab[: int(vocab_size)]

    vocab_size = len(vocab)
    print("Final vocab size = %d" % vocab_size)

    print("Most common words remaining:", " ".join(vocab[:10]))
    vocab.sort()

    fh.write_to_json(vocab, os.path.join(output_dir, train_prefix + ".vocab.json"))

    count_dtype = np.uint16 if max_doc_length < np.iinfo(np.uint16).max else np.int

    train_X_sage, tr_aspect, tr_no_aspect, tr_widx, vocab_for_sage = process_subset(
        train_items,
        train_ids,
        train_parsed,
        train_labels,
        label_fields,
        label_lists,
        vocab,
        output_dir,
        train_prefix,
        count_dtype=count_dtype,
    )
    if n_test > 0:
        test_X_sage, te_aspect, te_no_aspect, _, _ = process_subset(
            test_items,
            test_ids,
            test_parsed,
            test_labels,
            label_fields,
            label_lists,
            vocab,
            output_dir,
            test_prefix,
            count_dtype=count_dtype,
        )

    train_sum = np.array(train_X_sage.sum(axis=0))
    print("%d words missing from training data" % np.sum(train_sum == 0))

    if n_test > 0:
        test_sum = np.array(test_X_sage.sum(axis=0))
        print("%d words missing from test data" % np.sum(test_sum == 0))

    print("Done!")

''' scipyの方でメモリエラーが出るので以下のコードは使わない。ラベルなしデータを学習する分には多分問題ないはず。
    sage_output = {
        "tr_data": train_X_sage,
        "tr_aspect": tr_aspect,
        "widx": tr_widx,
        "vocab": vocab_for_sage,
    }
    if n_test > 0:
        sage_output["te_data"] = test_X_sage
        sage_output["te_aspect"] = te_aspect
    savemat(os.path.join(output_dir, "sage_labeled.mat"), sage_output)
    sage_output["tr_aspect"] = tr_no_aspect
    if n_test > 0:
        sage_output["te_aspect"] = te_no_aspect
    savemat(os.path.join(output_dir, "sage_unlabeled.mat"), sage_output)
'''


# to pass to pool.imap
def _process_item(item, **kwargs):
    text = item["text"]
    label_fields = kwargs.pop("label_fields", None)
    labels = None
    if label_fields:
        # TODO: probably don't want blind str conversion here
        labels = {label_field: str(item[label_field]) for label_field in label_fields}
    if text:
        tokens, _ = tokenize(text, **kwargs)
    else:
        tokens = []
    return item.get("id", None), tokens, labels


def process_subset(
    items,
    ids,
    parsed,
    labels,
    label_fields,
    label_lists,
    vocab,
    output_dir,
    output_prefix,
    count_dtype=np.int,
):
    n_items = len(items)
    vocab_size = len(vocab)
    vocab_index = dict(zip(vocab, range(vocab_size)))

    if not ids or len(ids) != n_items:
        ids = [str(i) for i in range(n_items)]

    # create a label index using string representations
    if labels:
        labels_df = pd.DataFrame.from_records(labels, index=ids)

        for label_field in label_fields:
            labels_df_subset = pd.get_dummies(labels_df[label_field])

            # for any classes not present in the subset, add 0 columns
            # (handles case where classes appear in only one of train or test)
            for category in label_lists[label_field]:
                if category not in labels_df_subset:
                    labels_df_subset[category] = 0

            labels_df_subset.to_csv(
                os.path.join(output_dir, output_prefix + "." + label_field + ".csv")
            )
            if labels_df[label_field].nunique() == 2:
                labels_df_subset.iloc[:, 1].to_csv(
                    os.path.join(
                        output_dir, output_prefix + "." + label_field + "_vector.csv"
                    ),
                    header=[label_field],
                )
            # used later
            label_index = dict(
                zip(labels_df_subset.columns, range(len(labels_df_subset)))
            )
    if len(label_fields) > 0:
        X = np.zeros([n_items, vocab_size], dtype=count_dtype)
    else:
        X = sparse.lil_matrix((n_items, vocab_size), dtype=count_dtype)
    
    dat_strings = []
    dat_labels = []
    mallet_strings = []
    fast_text_lines = []

    counter = Counter()
    word_counter = Counter()
    doc_lines = []
    print("Converting to count representations")
    for i, words in enumerate(parsed):
        # get the vocab indices of words that are in the vocabulary
        words = words.split()

        indices = [vocab_index[word] for word in words if word in vocab_index]
        word_subset = [word for word in words if word in vocab_index]

        counter.clear()
        counter.update(indices)
        word_counter.clear()
        word_counter.update(word_subset)

        if len(counter.keys()) > 0:
            # udpate the counts
            mallet_strings.append(str(i) + "\t" + "en" + "\t" + " ".join(word_subset))

            dat_string = str(int(len(counter))) + " "
            dat_string += " ".join(
                [
                    str(k) + ":" + str(int(v))
                    for k, v in zip(list(counter.keys()), list(counter.values()))
                ]
            )
            dat_strings.append(dat_string)

            # for dat formart, assume just one label is given
            if len(label_fields) > 0:
                label = labels[i][label_fields[-1]]
                dat_labels.append(str(label_index[str(label)]))
            values = np.array(list(counter.values()), dtype=count_dtype)
            X[
                np.ones(len(counter.keys()), dtype=int) * i, list(counter.keys())
            ] += values

    # convert to a sparse representation
    if len(label_fields) > 0:
        sparse_X = sparse.csr_matrix(X)
    else:
        sparse_X = X.tocsr()

    fh.save_sparse(sparse_X, os.path.join(output_dir, output_prefix + ".npz"))

    print("Size of {:s} document-term matrix:".format(output_prefix), sparse_X.shape)

    fh.write_to_json(ids, os.path.join(output_dir, output_prefix + ".ids.json"))

    # save output for Mallet
    fh.write_list_to_text(
        mallet_strings, os.path.join(output_dir, output_prefix + ".mallet.txt")
    )

    # save output for David Blei's LDA/SLDA code
    fh.write_list_to_text(
        dat_strings, os.path.join(output_dir, output_prefix + ".data.dat")
    )
    if len(dat_labels) > 0:
        fh.write_list_to_text(
            dat_labels,
            os.path.join(output_dir, output_prefix + "." + label_field + ".dat"),
        )

    # save output for Jacob Eisenstein's SAGE code:
    if len(label_fields) > 0:
        sparse_X_sage = sparse.csr_matrix(X, dtype=float)
    else:
        sparse_X_sage = X.tocsr()

    vocab_for_sage = np.zeros((vocab_size,), dtype=np.object)
    vocab_for_sage[:] = vocab

    # for SAGE, assume only a single label has been given
    if len(label_fields) > 0:
        # convert array to vector of labels for SAGE
        sage_aspect = (
            np.argmax(np.array(labels_df_subset.values, dtype=float), axis=1) + 1
        )
    else:
        sage_aspect = np.ones([n_items, 1], dtype=float)
    sage_no_aspect = np.array([n_items, 1], dtype=float)
    widx = np.arange(vocab_size, dtype=float) + 1

    return sparse_X_sage, sage_aspect, sage_no_aspect, widx, vocab_for_sage


def tokenize(
    text,
    strip_html=False,
    lower=True,
    keep_emails=False,
    keep_at_mentions=False,
    keep_numbers=False,
    keep_alphanum=False,
    min_length=3,
    stopwords=None,
    ngram_range=(1, 1),
    vocab=None,
):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)
    tokens = text.split()

    if stopwords is not None:
        tokens = ["_" if t in stopwords else t for t in tokens]

    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:
        tokens = [t if alpha.match(t) else "_" for t in tokens]

    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t if alpha_or_num.match(t) else "_" for t in tokens]

    # drop short tokens
    if min_length > 0:
        tokens = [t if len(t) >= min_length else "_" for t in tokens]

    counts = Counter()

    unigrams = [t for t in tokens if t != "_"]
    counts.update(unigrams)

    if vocab is not None:
        tokens = [token for token in unigrams if token in vocab]
    else:
        tokens = unigrams

    tokens = [
        "_".join(tokens[j : j + i])
        for i in range(min(ngram_range), max(ngram_range) + 1)
        for j in range(len(tokens) - i + 1)
    ]

    return tokens, counts


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
    text = re.sub(r"\.", "", text)
    # replace all other punctuation (except single quotes) with spaces
    text = replace.sub(" ", text)
    # remove single quotes
    text = re.sub(r"\'", "", text)
    # replace all whitespace with a single space
    text = re.sub(r"\s", " ", text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == "__main__":
    main(sys.argv)

