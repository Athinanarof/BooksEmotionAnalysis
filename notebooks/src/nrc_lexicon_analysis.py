import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm_notebook as tqdm
from tqdm import trange


import os
from pathlib import Path

from src import my_books_sentiment_analyzer as my_analyzer

def text_emotion(df, column):
    '''
    Takes a DataFrame and a specified column of text and adds 10 columns to the
    DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
    column containing the value of the text in that emotions
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns
    '''

    new_df = df.copy()

    filepath = ('../data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    emolex_df = pd.read_csv(filepath,
                            names=["word", "emotion", "association"],
                            sep='\t')
    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='association').reset_index()
    emotions = emolex_words.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

    stemmer = SnowballStemmer("english")


    book = ''
    chapter = ''

    with tqdm(total=len(list(new_df.iterrows()))) as pbar:
        for i, row in new_df.iterrows():
            pbar.update(1)
            if row['book'] != book:
                print(row['book'])
                book = row['book']
            if row['chapter_title'] != chapter:
                print('   ', row['chapter_title'])
                chapter = row['chapter_title']
                chap = row['chapter_title']
            document = word_tokenize(new_df.loc[i][column])
            for word in document:
                word = stemmer.stem(word.lower())
                emo_score = emolex_words[emolex_words.word == word]
                if not emo_score.empty:
                    for emotion in list(emotions):
                        emo_df.at[i, emotion] += emo_score[emotion]

    new_df = pd.concat([new_df, emo_df], axis=1)

    return new_df



def get_books_data(my_books_dictionary):
    data = {'book': [], 'chapter_title': [], 'text': []}

    for book in my_books_dictionary:
        print(book)
        for chapter in tqdm(my_books_dictionary[book]):
            chapter =  my_books_dictionary[book][chapter]
            title = chapter[0]
            text = chapter[1].replace('\n', '')

            data['book'].append(book)
            data['chapter_title'].append(title)
            data['text'].append(text)

    books_df = pd.DataFrame(data=data)
    return books_df

def print_chapters_emotion(dataframe, my_books_dictionary):
    for emotion in emotions:
        for book in my_books_dictionary:
            for chapter in my_books_dictionary[book]:
                print(book)

                book_data = my_books_dictionary[book]
                print(book_data[chapter][0])
                print(dataframe.loc[book].loc[book_data[chapter][0]][emotion])
                print('\n')
# plotting

def get_from_dataframe_books_information(dataframe, my_books_dictionary, emotion):
    y = [dataframe.loc[book].loc[my_books_dictionary[book][chapter][0]][emotion] for book in my_books_dictionary for chapter in my_books_dictionary[book]]
    return y

def get_total_length_of_book(my_books_dictionary):
    return  sum([len(my_books_dictionary[book]) for book in my_books_dictionary])


def plot_all_analysis_of_all_emotions(dataframe, my_books_dictionary, book_indices, emotions):
    length = get_total_length_of_book(my_books_dictionary)
    x = np.linspace(0, length - 1, num=length)

    for emotion in emotions:
        y = get_from_dataframe_books_information(dataframe, my_books_dictionary, emotion)

        plt.figure(figsize=(15, 10))
        for book in book_indices:
            #print(book)
            book_indices_values = book_indices[book]
            plt.plot(x[book_indices_values[0]: book_indices_values[1]],
                     y[book_indices_values[0]: book_indices_values[1]],
                     label=book)
        plt.plot(my_analyzer.movingaverage(y, 10), color='k', linewidth=3, linestyle=':', label='Moving Average')
        plt.legend(loc='best', fontsize=15, bbox_to_anchor=(1.05, 1))
        plt.title('{} Sentiment of the entire list of books'.format(emotion.title()), fontsize=20)
        plt.xlabel('Chapter', fontsize=15)
        plt.ylabel('Average Sentiment', fontsize=15)
        plt.show()


def plot_all_emotions_analysis_in_single_figure(dataframe, my_books_dictionary, book_indices, emotions):
    length = get_total_length_of_book(my_books_dictionary)
    x = np.linspace(0, length - 1, num=length)

    fig, ax = plt.subplots(4, 3, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.1)
    fig.suptitle('Sentiment of the entire list of books', fontsize=20, y=1.02)
    fig.subplots_adjust(top=0.88)

    ax = ax.ravel()

    for i, emotion in enumerate(emotions):
        y = get_from_dataframe_books_information(dataframe, my_books_dictionary, emotion)
        for book in book_indices:

            book_indices_values = book_indices[book]
            ax[i].plot(x[book_indices_values[0]: book_indices_values[1]],
                     y[book_indices_values[0]: book_indices_values[1]],
                     label=book, linewidth=2)

        ax[i].set_title('{} Sentiment'.format(emotion.title()))
        ax[i].set_xticks([])

    fig.legend(list(my_books_dictionary), loc='upper right', fontsize=15, bbox_to_anchor=(.85, .2))
    fig.tight_layout()
    fig.delaxes(ax[-1])
    fig.delaxes(ax[-2])
    plt.show()

tab10 = matplotlib.cm.get_cmap('tab10')

def create_plot_stack(x, y, emotions):
    return  plt.stackplot(x, y, colors=(tab10(0),
                                    tab10(.1),
                                    tab10(.2),
                                    tab10(.3),
                                    tab10(.4),
                                    tab10(.5),
                                    tab10(.6),
                                    tab10(.7),
                                    tab10(.8),
                                    tab10(.9)), labels=emotions)


def common_plot_information(fig, ax, book_indices, my_books_dictionary):
    # Plot vertical lines marking the books
    for book in book_indices:
        plt.axvline(x=book_indices[book][0], color='black', linewidth=3, linestyle=':')

    plt.axvline(x=book_indices[book][1]-1, color='black', linewidth=3, linestyle=':')

    common_plot_legends(book_indices, my_books_dictionary)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best', fontsize=15, bbox_to_anchor=(1.2, 1))

    return (fig, ax)

def common_plot_legends(book_indices, my_books_dictionary):
    plt.title('Emotional Sentiment of the Entire List of Books', fontsize=20)
    plt.xticks([(book_indices[book][0] + book_indices[book][1]) / 2 for book in book_indices],
               list(my_books_dictionary),
               rotation=-30,
               fontsize=15,
               ha='left')
    plt.yticks([])
    plt.ylabel('Relative Sentiment', fontsize=15)

def plot_A1(dataframe, my_books_dictionary, book_indices, emotions):
    length = get_total_length_of_book(my_books_dictionary)
    x = np.linspace(0, length - 1, num=length)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    y = [dataframe[emotion].tolist() for emotion in emotions]

    create_plot_stack(x, y, emotions)
    common_plot_information(fig, ax, book_indices, my_books_dictionary)

    ax.grid(False)

    plt.show()



def plot_A2(dataframe, my_books_dictionary, book_indices, emotions):
    length = get_total_length_of_book(my_books_dictionary)
    x = np.linspace(0, length - 1, num=length)
    window = 10

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    y = [my_analyzer.movingaverage(dataframe[emotion].tolist(), window) for emotion in emotions]

    create_plot_stack(x, y, emotions)
    common_plot_information(fig, ax, book_indices, my_books_dictionary)

    axes = plt.gca()
    axes.set_xlim([min(x) + window / 2, max(x) - window / 2])

    ax.grid(False)

    plt.show()

def plot_A3(dataframe, my_books_dictionary, book_indices, emotions):
    length = get_total_length_of_book(my_books_dictionary)
    window = 10

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    x = np.linspace(0, length - 1, num=length)[int(window / 2): -int(window / 2)]
    y = [my_analyzer.movingaverage(dataframe[emotion].tolist(), window)[int(window / 2): -int(window / 2)] for emotion in emotions]

    create_plot_stack(x, y, emotions)
    common_plot_information(fig, ax, book_indices, my_books_dictionary)

    ax.grid(False)

    plt.show()


def plot_A4(dataframe, my_books_dictionary, book_indices, emotions):
    length =  get_total_length_of_book(my_books_dictionary)
    x = np.linspace(0, length - 1, num=length)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    for emotion in emotions:
        y = get_from_dataframe_books_information(dataframe, my_books_dictionary, emotion)
        plt.plot(x, y, linewidth=2, label=emotion)

    common_plot_information(fig, ax, book_indices, my_books_dictionary)
    plt.legend(loc='best', fontsize=15, bbox_to_anchor=(1.2, 1))
    plt.xlabel('Chapter', fontsize=15)
    plt.ylabel('Average Sentiment', fontsize=15)
    plt.show()


def plot_A5(dataframe, my_books_dictionary, book_indices, emotions):
    length =  get_total_length_of_book(my_books_dictionary)
    window = 20

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    x = np.linspace(0, length - 1, num=length)[int(window / 2): -int(window / 2)]

    for c, emotion in enumerate(emotions):
        y = get_from_dataframe_books_information(dataframe, my_books_dictionary, emotion)
        y = my_analyzer.movingaverage(y, window)[int(window / 2): -int(window / 2)]
        plt.plot(x, y, linewidth=5, label=emotion, color=(tab10(c)))

    plt.legend(loc='best', fontsize=15, bbox_to_anchor=(1.2, 1))
    common_plot_information(fig, ax, book_indices, my_books_dictionary)

    ax.grid(False)

    plt.show()

def plot_two_opposite_emotions(dataframe, my_books_dictionary, book_indices, emotions):
    length =  get_total_length_of_book(my_books_dictionary)
    x = np.linspace(0, length - 1, num=length)

    plt.figure(figsize=(15, 10))
    for emotion in emotions:
        y = get_from_dataframe_books_information(dataframe, my_books_dictionary, emotion)
        plt.plot(x, y, linewidth=2, label=emotion)

    plt.legend(loc='best', fontsize=15)
    plt.title('Emotional Sentiment of the entire list of books', fontsize=20)
    plt.xlabel('Chapter', fontsize=15)
    plt.ylabel('Average Sentiment', fontsize=15)
    plt.show()


def plot_two_opposite_emotions_display_area(dataframe, my_books_dictionary, book_indices, emotions):
    length =  get_total_length_of_book(my_books_dictionary)
    x = np.linspace(0, length - 1, num=length)

    plt.figure(figsize=(15, 15))
    y = [dataframe[emotion].tolist() for emotion in emotions]
    plt.stackplot(x, y, labels=emotions)

    for book in book_indices:
        plt.axvline(x=book_indices[book][1], color='black', linewidth=3, linestyle=':')

    plt.legend(loc='best', fontsize=15, bbox_to_anchor=(1.2, 1))
    common_plot_legends(book_indices, my_books_dictionary)
    plt.ylabel('Average Sentiment', fontsize=15)
    plt.show()

def plot_analysis_grouped_by_book(dataframe, my_books_dictionary, book_indices, emotions):
    length = len(my_books_dictionary)
    x = np.linspace(0, length - 1, num=length)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1,1)

    y = [dataframe[emotion].tolist() for emotion in emotions]
    plt.stackplot(list(my_books_dictionary), y, labels=emotions)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best', fontsize=15, bbox_to_anchor=(1.2, 1))
    plt.title('Emotional Sentiment of the entire list of books', fontsize=20)
    plt.xticks(list(my_books_dictionary), list(my_books_dictionary), rotation=-30, ha='left', fontsize=15)
    plt.ylabel('Relative Sentiment Score', fontsize=15)
    plt.yticks([])
    ax.grid(False)
    plt.show()

def plot_A6(dataframe, my_books_dictionary, book_indices, emotions):
    books = list(my_books_dictionary)
    margin_bottom = np.zeros(len(my_books_dictionary))

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    for c, emotion in enumerate(emotions):
        y = np.array(dataframe[emotion])
        plt.bar(books, y, bottom=margin_bottom, label=emotion, color=(tab10(c)))
        margin_bottom += y

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best', fontsize=15, bbox_to_anchor=(1.2, 1))
    plt.title('Emotional Sentiment of the entire list of book', fontsize=20)
    plt.xticks(books, books, rotation=-30, ha='left', fontsize=15)
    plt.ylabel('Relative Sentiment Score', fontsize=15)
    plt.yticks([])
    ax.grid(False)
    plt.show()

def plot_A7(dataframe, my_books_dictionary, book_indices, emotions):
    books = list(my_books_dictionary)
    margin_bottom = np.zeros(len(books))

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    for c, emotion in enumerate(emotions):
        y = np.array(dataframe[emotion])
        plt.bar(books, y, bottom=margin_bottom, label=emotion) #emotions[len(emotions) - idx - 1])
        margin_bottom += y

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best', fontsize=15, bbox_to_anchor=(1.2, 1))
    plt.title('Emotional Sentiment of the entire list of book', fontsize=20)
    plt.xticks(books, books, rotation=-30, ha='left', fontsize=15)
    plt.ylabel('Relative Sentiment Score', fontsize=15)
    plt.yticks([])
    ax.grid(False)
    plt.show()
