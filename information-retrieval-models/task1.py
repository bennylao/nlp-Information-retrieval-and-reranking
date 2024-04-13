import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import ssl


def process_raw_text(lines, remove_stopwords=False, lemmatisation=False, stemming=False):
    tokens = []
    if remove_stopwords is True:
        # load stopwords into variable
        stop_words = nltk.corpus.stopwords.words('english')
    if lemmatisation is True:
        nltk.download('wordnet')
        # load lemmatiser into variable
        lemmatiser = WordNetLemmatizer()
    if stemming is True:
        # load stemmer into variable
        stemmer = SnowballStemmer("english")

    for line in lines:
        # make words lowercase
        line = line.strip().lower()

        # replace anything that is not A-Z or a-z or space with a space
        line = re.sub(r"[^a-zA-Z\s]", " ", line)

        # tokenise using nltk
        words = [w for w in nltk.word_tokenize(line)]
        if remove_stopwords is True:
            words = [w for w in words if w not in stop_words]
        if lemmatisation is True:
            words = [lemmatiser.lemmatize(w) for w in words]
        if stemming is True:
            words = [stemmer.stem(w) for w in words]
        tokens.append(words)

    return tokens


def count_vocab(tokens):
    # load stopwords into variable
    stop_words_list = nltk.corpus.stopwords.words('english')

    tokens_flatten = [token for sublist in tokens for token in sublist]
    s = pd.Series(tokens_flatten).value_counts()
    df = s.reset_index()
    df.columns = ["vocab", "count"]
    mask = ~df['vocab'].isin(stop_words_list)
    df_remove_stopwords = df.loc[mask, :]
    df.index = np.arange(1, df.shape[0] + 1)
    df_remove_stopwords.index = np.arange(1, df_remove_stopwords.shape[0] + 1)
    df['norm_freq'] = (df['count'] / df['count'].sum())
    df_remove_stopwords['norm_freq'] = (df_remove_stopwords['count'] / df_remove_stopwords['count'].sum())

    return df, df_remove_stopwords


def calculate_zipf_norm_freq(df):
    denominator = sum(1 / i for i in df.index)

    return 1 / (df.index * denominator)


def plot_graph(df, df_no_stopwords):
    zipf_norm_freq = calculate_zipf_norm_freq(df)

    # figure 1
    plt.figure(constrained_layout=True)
    plt.plot(df.index, zipf_norm_freq, '--', label="Zipf's Distribution")
    plt.plot(df.index, df['norm_freq'], '-', label="Empirical Distribution")
    plt.xlabel("Term's Frequency Ranking", fontsize="x-large")
    plt.ylabel("Normalised Frequency", fontsize="x-large")
    plt.legend()
    plt.savefig('Figure_1.pdf')

    # figure 2
    plt.figure(constrained_layout=True)
    plt.loglog(df.index, zipf_norm_freq, '-', label="Zipf's Distribution")
    plt.loglog(df.index, df['norm_freq'], '--', label="Empirical Distribution")
    plt.xlabel(r"$Term's \ Frequency \ Ranking \ (log_{10})$", fontsize="x-large")
    plt.ylabel(r"$Normalised \ Frequency \ (log_{10})$", fontsize="x-large")
    plt.legend()
    plt.savefig('Figure_2.pdf')

    # figure 3
    plt.figure(constrained_layout=True)
    plt.loglog(df.index, zipf_norm_freq, '-', label="Zipf's Distribution")
    plt.loglog(df_no_stopwords.index, df_no_stopwords['norm_freq'], '--', label="Empirical Distribution")
    plt.xlabel(r"$Term's \ Frequency \ Ranking \ (log_{10})$", fontsize="x-large")
    plt.ylabel(r"$Normalised \ Frequency \ (log_{10})$", fontsize="x-large")
    plt.legend()
    plt.savefig('Figure_3.pdf')


if __name__ == "__main__":
    # if SSL certificate verify fails (probably eduroam wifi has blocked this kind of connection),
    # try using alternative wifi or use the try-except code below.
    # reference: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    start = timer()

    pd.options.mode.copy_on_write = True

    # download list of stopwords from nltk
    nltk.download('stopwords')
    with open("passage-collection.txt", encoding="utf8", mode="r") as file:
        raw_text = file.readlines()

    tokens_list = process_raw_text(raw_text, remove_stopwords=False, stemming=False, lemmatisation=False)
    df_vocab, df_vocab_no_stopwords = count_vocab(tokens_list)
    plot_graph(df_vocab, df_vocab_no_stopwords)

    end = timer()
    time_taken = end - start

    print(f"Task 1 Number of tokens: {df_vocab['count'].sum()}")
    print(f"Task 1 Number of vocabulary: {df_vocab.shape[0]}")
    print(f"Task 1 Number of tokens (no stopwords): {df_vocab_no_stopwords['count'].sum()}")
    print(f"Task 1 Number of vocabulary (no stopwords): {df_vocab_no_stopwords.shape[0]}")
    print(f"Task 1 Process time: {time_taken} seconds")  # usually takes 25 sec
