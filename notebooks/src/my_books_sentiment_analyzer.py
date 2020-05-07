import re
import pandas as pd
import numpy as np
from collections import defaultdict

# Set Pandas to display all rows of dataframes
pd.set_option('display.max_rows', 500)

# nltk
from nltk import tokenize

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Plotting tools
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm_notebook as tqdm
from tqdm import trange

import os
from pathlib import Path

import nltk
nltk.download('punkt')


def get_books_directory_path():
    return Path('../data/my_books').resolve()


def get_list_of_books(directory_path, books):

    directory_files = os.listdir(directory_path)

    with tqdm(total=len(directory_files)) as t:
        for filename in directory_files:
            books.append(filename)

    print(books)

    return books


def define_pattern_to_get_chapters_information():
    return ("((?:CHAPTER[ ]|Chapter[ ])(?:[A-Za-z]+|[0-9]+))\n+" +
                                    "([A-Za-z',. -]+)\\b(?![A-za-z]+(?=\.))\\b" +
                                    "(?![a-z']|[A-Z.])" +
                                    "(.*?)" +
                                    "(?=CHAPTER|Chapter (?:[A-Za-z]+|[0-9]+)|This book \n|- THE END -)")

def initialize_books_dictionary(directory_path, my_books_dictionary, books, books_elements_pattern):

    with tqdm(total=len(books)) as t:
        for book in books:
            title = book[:-4]

            with open(os.path.join(directory_path, book), 'r') as file:
                text = (file.read().replace('&rsquo;', "'")
                                    .replace('&lsquo;', "'")
                                    .replace('&rdquo;', '"')
                                    .replace('&ldquo;', '"')
                                    .replace('&mdash;', '—'))
                chapters = re.findall(books_elements_pattern, text, re.DOTALL)
                chapter_number = 0

                for chapter in chapters:
                    chapter_number += 1
                    chapter_title = chapter[0] + ' ' + chapter[1].replace('\n', '')
                    chapter_text = chapter[2]

                    chapter_text = re.sub('\n*&bull; [0-9]+ &bull; \n*' + chapter_title + ' \n*', '', chapter_text, flags=re.IGNORECASE)
                    chapter_text = re.sub('\n*&bull; [0-9]+ &bull;\s*(CHAPTER [A-Z-]+\s*)|(EPILOGUE)\s*', '', chapter_text)
                    chapter_text = re.sub(' \n&bull; [0-9]+ &bull; \n*', '', chapter_text)
                    chapter_text = re.sub('\s*'.join([word for word in chapter_title.split()]), '', chapter_text)

                    my_books_dictionary[title]['Chapter' + str(chapter_number)] = (chapter_title, chapter_text)
        my_books_dictionary = dict(my_books_dictionary)

    return my_books_dictionary

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_analysis_by_average_sentences_value_in_chapter(my_books_dictionary):

    for book in tqdm(my_books_dictionary, desc="Progress"):
        print(book)
        for chapter in tqdm(my_books_dictionary[book], postfix=book):
            print('  ', my_books_dictionary[book][chapter][0])
            text = my_books_dictionary[book][chapter][1].replace('\n', '')
            sentence_list = tokenize.sent_tokenize(text)
            sentiments = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0 }

            for sentence in sentence_list:
                vs = analyzer.polarity_scores(sentence)
                sentiments['compound'] += vs['compound']
                sentiments['neg'] += vs['neg']
                sentiments['neu'] += vs['neu']
                sentiments['pos'] += vs['pos']

            sentiments['compound'] = sentiments['compound'] / len(sentence_list)
            sentiments['neg'] = sentiments['neg'] / len(sentence_list)
            sentiments['neu'] = sentiments['neu'] / len(sentence_list)
            sentiments['pos'] = sentiments['pos'] / len(sentence_list)

            my_books_dictionary[book][chapter] = (my_books_dictionary[book][chapter][0], my_books_dictionary[book][chapter][1], sentiments)

    return my_books_dictionary

def get_sentiment_analysis_from_entire_chapter(my_books_dictionary):

    for book in tqdm(my_books_dictionary, desc='Progress'):
        print(book)
        for chapter in tqdm(my_books_dictionary[book]):
            text = my_books_dictionary[book][chapter][1].replace('\n', '')
            sentence_list = tokenize.sent_tokenize(text)
            sentiments = {'compound': [], 'neg': [], 'neu': [], 'pos': []}

            vs = analyzer.polarity_scores(text)
            sentiments['compound'] = vs['compound']
            sentiments['neg'] = vs['neg']
            sentiments['neu'] = vs['neu']
            sentiments['pos'] = vs['pos']

            my_books_dictionary[book][chapter] = (my_books_dictionary[book][chapter][0], my_books_dictionary[book][chapter][1], sentiments)
    return my_books_dictionary


def get_compount_sentiments(my_books_dictionary):
    return [my_books_dictionary[book][chapter][2]['compound'] for book in my_books_dictionary for chapter in my_books_dictionary[book]]

def print_books_chapters_compound_sentiment(my_books_dictionary, compound_sentiments):
    chap = 0
    for book in my_books_dictionary:
        print(book)
        book_chap = 1
        for chapter in my_books_dictionary[book]:
            print('  Chapter', book_chap, '-', my_books_dictionary[book][chapter][0])
            print('      ', compound_sentiments[chap])
            book_chap += 1
            chap += 1
        print()

def get_books_indices(my_books_dictionary):
    book_indices = {}
    idx = 0
    for book in my_books_dictionary:
        start = idx
        for chapter in my_books_dictionary[book]:
            idx += 1
        book_indices[book] = (start, idx)
    return book_indices

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def plot_books_chapters_sentiment(my_books_dictionary, compound_sentiments, book_indices):

    length= sum([len(my_books_dictionary[book]) for book in my_books_dictionary])
    x = np.linspace(0, length - 1, num=length)
    y = compound_sentiments


    plt.figure(figsize=(20, 10))

    for book in book_indices:
        tuple_values = book_indices[book]
        plt.plot(x[tuple_values[0]: tuple_values[1]],
                 y[tuple_values[0]: tuple_values[1]],
                 label=book)

    plt.plot(movingaverage(y, 10), color='k', linewidth=3, linestyle=':', label = 'Moving Average')
    plt.axhline(y=0, xmin=0, xmax=length, alpha=.25, color='r', linestyle='--', linewidth=3)
    plt.legend(loc='best', fontsize=15)
    plt.title("Sentiment of the books' chapters", fontsize=20)
    plt.xlabel('Chapter', fontsize=15)
    plt.ylabel('Average Sentiment', fontsize=15)
    plt.show()

def get_chapters_sentiment_scores_values(my_books_dictionary):
    return [[my_books_dictionary[book][chapter][2][sentiment] for book in my_books_dictionary for chapter in my_books_dictionary[book]]
                    for sentiment in ['compound', 'neg', 'neu', 'pos']]

def print_average_books_sentiment(sentiment_scores, book_indices):

    compound_sentiment = sentiment_scores[0]

    print('Average Book Sentiment:')
    print()

    for book in book_indices:
        tuple_values = book_indices[book]
        compound = compound_sentiment[tuple_values[0]: tuple_values[1]]

        print('{:45}{:.2f}%'.format(book, 100 * sum(compound) / len(compound)))


def plot_sentiment_of_entire_serie_of_books(my_books_dictionary, sentiment_scores):
    length = sum([len(my_books_dictionary[book]) for book in my_books_dictionary])
    x = np.linspace(0, length - 1, num=length)

    plt.figure(figsize=(15, 10))
    for i, sentiment in enumerate(sentiment_scores):
        plt.plot(x,
                 sentiment,
                 label=['compound', 'neg', 'neu', 'pos'][i])
    # plt.plot(movingaverage(compound_sentiments, 10)+.1, color='k', linewidth=3, linestyle=':', label = 'Moving Average')
    plt.axhline(y=0, xmin=0, xmax=length, alpha=.25, color='r', linestyle='--', linewidth=3)
    plt.legend(loc='best', fontsize=15)
    plt.title('Chapter Sentiment of the list of Books', fontsize=20)
    plt.xlabel('Chapter', fontsize=15)
    plt.ylabel('Average Sentiment', fontsize=15)
    plt.show()



# Functions for analysis comparision

def plot_compound_sentiment_comparison_between_analysis(dictionary_by_sentences, dictionary_by_whole_chapter):

    plt.figure(figsize=(15, 10))
    plt.plot([dictionary_by_sentences[book][chapter][2]['compound'] for book in dictionary_by_sentences for chapter in dictionary_by_sentences[book]])
    plt.plot([dictionary_by_whole_chapter[book][chapter][2]['compound'] for book in dictionary_by_whole_chapter for chapter in dictionary_by_whole_chapter[book]], linestyle="--")
    plt.title('Compound sentiment comparision between analysis', fontsize=20)

    analysis_labels = ['Analysis by sentence', 'Analysis by whole chapter']
    plt.legend(analysis_labels)
    plt.show()

def plot_negative_sentiment_comparison_between_analysis(dictionary_by_sentences, dictionary_by_whole_chapter):

    plt.figure(figsize=(15, 10))
    plt.plot([dictionary_by_sentences[book][chapter][2]['neg'] for book in dictionary_by_sentences for chapter in dictionary_by_sentences[book]])
    plt.plot([dictionary_by_whole_chapter[book][chapter][2]['neg'] for book in dictionary_by_whole_chapter for chapter in dictionary_by_whole_chapter[book]], linestyle="--")
    plt.title('Compound negative comparision between analysis', fontsize=20)

    analysis_labels = ['Analysis by sentence', 'Analysis by whole chapter']
    plt.legend(analysis_labels)
    plt.show()

def plot_neutral_sentiment_comparison_between_analysis(dictionary_by_sentences, dictionary_by_whole_chapter):

    plt.figure(figsize=(15, 10))
    plt.plot([dictionary_by_sentences[book][chapter][2]['neu'] for book in dictionary_by_sentences for chapter in dictionary_by_sentences[book]])
    plt.plot([dictionary_by_whole_chapter[book][chapter][2]['neu'] for book in dictionary_by_whole_chapter for chapter in dictionary_by_whole_chapter[book]], linestyle="--")
    plt.title('Compound neutral comparision between analysis', fontsize=20)

    analysis_labels = ['Analysis by sentence', 'Analysis by whole chapter']
    plt.legend(analysis_labels)
    plt.show()

def plot_positive_sentiment_comparison_between_analysis(dictionary_by_sentences, dictionary_by_whole_chapter):

    plt.figure(figsize=(15, 10))
    plt.plot([dictionary_by_sentences[book][chapter][2]['pos'] for book in dictionary_by_sentences for chapter in dictionary_by_sentences[book]])
    plt.plot([dictionary_by_whole_chapter[book][chapter][2]['pos'] for book in dictionary_by_whole_chapter for chapter in dictionary_by_whole_chapter[book]], linestyle="--")
    plt.title('Compound positive comparision between analysis', fontsize=20)

    analysis_labels = ['Analysis by sentence', 'Analysis by whole chapter']
    plt.legend(analysis_labels)
    plt.show()

def plot_all_sentiments_comparision_between_analysis(sentiment_scores_by_sentences, sentiment_scores_by_whole_chapter, dictionay_by_sentence, dictionary_by_whole_chapter):

    length = sum([len(dictionay_by_sentence[book]) for book in dictionay_by_sentence])
    x = np.linspace(0, length - 1, num=length)

    plt.figure(figsize=(15, 15))
    for i, sentiment in enumerate(sentiment_scores_by_sentences):
        plt.plot(x,
                 sentiment,
                 label=['compound', 'neg', 'neu', 'pos'][i])
    # plt.plot(movingaverage(compound_sentiments, 10)+.1, color='k', linewidth=3, linestyle=':', label = 'Moving Average')
    plt.axhline(y=0, xmin=0, xmax=length, alpha=.25, color='r', linestyle='--', linewidth=3)

    length = sum([len(dictionary_by_whole_chapter[book]) for book in dictionary_by_whole_chapter])
    x = np.linspace(0, length - 1, num=length)

    for i, sentiment in enumerate(sentiment_scores_by_whole_chapter):
        plt.plot(x,
                 sentiment,
                 label=['compound', 'neg', 'neu', 'pos'][i],
                 linestyle=':')
    plt.axhline(y=0, xmin=0, xmax=length, alpha=.25, color='r', linestyle='--', linewidth=3)
    plt.legend(loc='best', fontsize=15)
    plt.title('Chapter Sentiment of the Harry Potter series', fontsize=20)
    plt.xlabel('Chapter')
    plt.ylabel('Average Sentiment')
    plt.show()


def plot_all_sentiments_comparision_between_analysis_except_compound(sentiment_scores_by_sentences, sentiment_scores_by_whole_chapter, dictionay_by_sentence, dictionary_by_whole_chapter):

    length = sum([len(dictionay_by_sentence[book]) for book in dictionay_by_sentence])
    x = np.linspace(0, length - 1, num=length)

    plt.figure(figsize=(15, 15))
    for i, sentiment in enumerate(sentiment_scores_by_sentences[1:]):
        plt.plot(x,
                 sentiment,
                 label=['neg', 'neu', 'pos'][i])

    length = sum([len(dictionary_by_whole_chapter[book]) for book in dictionary_by_whole_chapter])
    x = np.linspace(0, length - 1, num=length)

    for i, sentiment in enumerate(sentiment_scores_by_whole_chapter[1:]):
        plt.plot(x,
                 sentiment,
                 label=['neg', 'neu', 'pos'][i],
                 linestyle=':')
    plt.legend(loc='best', fontsize=15)
    plt.show()
