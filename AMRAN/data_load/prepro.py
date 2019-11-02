import os
import re
import nltk
import json

import numpy as np
from datetime import datetime
from collections import Counter
from tqdm import tqdm


def word_tokenize(tokens, lower=True):
    if lower:
        return [token.replace("''", '"').replace("``", '"').lower() for token in nltk.word_tokenize(tokens)]
    else:
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def remove_emoji(temp_tokens):
    emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', temp_tokens)


def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-","\u2026" , "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


# not use
def get_glove(glove_path):
    glove_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        count = 0
        for line in fh:
            try:
                count += 1
                array = line.lstrip().rstrip().split(' ')
                word = array[0]
                vector = list(map(float, array[1:]))
                glove_dict[word] = list(map(float, array[1:]))
            except Exception as e:
                print (line, count)
            
    return glove_dict


# process word
def get_word2vec(glove_path, word_counter, limit=-1):
    word2vec_dict = {}
    word2idx_dict = {}
    word_embed_mat = []
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            array = line.lstrip().rstrip().split(' ')
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter and word_counter[word] > limit:
                word2vec_dict[word] = vector
            #elif word.capitalize() in word_counter:
            #    word2vec_dict[word.capitalize()] = vector
            #elif word.lower() in word_counter and and word_counter[word] > limit:
            #    word2vec_dict[word.lower()] = vector
            #elif word.upper() in word_counter:
            #    word2vec_dict[word.upper()] = vector
    #NULL = '--NULL--'
    OOV = '--OOV--'
    word2idx_dict = {word: idx for (idx, word) in enumerate(word2vec_dict.keys(), 1)}
    word2idx_dict[OOV] = 0

    default_vec = [0 for i in range(len(vector))]
    idx2emb_dict = {idx: word2vec_dict.get(word, default_vec) for (word, idx) in word2idx_dict.items()}
    word_embed_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]

    print('{}/{} of word vocab have corresponding vectors in {}, total {}'.format(len(word2vec_dict), len(word_counter), 'glove', len(idx2emb_dict)))
    return word_embed_mat, word2idx_dict


# process char
def char2idx(char_counter, limit=-1):
    char_dict = {k: v for (k, v) in char_counter.items() if v > limit}
    char2idx_dict = {char: idx for (idx, char) in enumerate(char_dict.keys(), 1)}
    char2idx_dict['--OOV--'] = 0 
    print('char total num {}, total {}'.format((len(char2idx_dict) - 1), len(char2idx_dict)))
    return char2idx_dict


# get tweets and url content word/char infomation
# use function: get_word2vec, char2idx
# source file: mapped_url_content.txt, mapped_tweet_content.txt
def get_word_char_info(glove_path, source_dir, data_types=['url', 'tweet'], limit=-1):
    sent_tokenize = nltk.sent_tokenize
    word_counter, char_counter = Counter(), Counter()
    for data_type in data_types:
        print('Processing {} file...'.format(data_type))
        source_path = os.path.join(source_dir, 'mapped_{}_content.txt'.format(data_type))
        with open(source_path) as f:
            total_lines = len(f.readlines())
        source_data = open(source_path, 'r')

        # TWEETS: source_data = {data: [{uid:xxxx, name:xxxx, content:[{text:xxxx}, {text:xxxx},...,{text:xxxx}]},{},...,{}], data_type: tweets}
        # URLS: source_data = {data: [{url_id:xxxx, url:xxxxx, content:[{text:xxxx}]},{},...,{}], data_type: urls}   
        for line in source_data: #tqdm(source_data, total=total_lines):
            raw_data = json.loads(line)
            data = raw_data['data']
            assert len(data.keys()) == 3

            if data['content'] is None:
                continue

            for cont in data['content']:
                context = cont['text']
                context = context.encode('ascii', 'ignore').decode('utf-8')
                #words
                context = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', context)
                context = re.sub(r'[a-z]*[:.]+\S+', '', context)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')
                context = remove_emoji(context)

                #print(context)
                # eg. 'I have an apple.'
                xi = list(map(word_tokenize, sent_tokenize(context)))
                xi = [process_tokens(tokens) for tokens in xi] # [['I', 'have', 'an', 'apple', '.']]
                cxi = [[list(xijk) for xijk in xij] for xij in xi] # [[['I'], ['h', 'a', 'v', 'e'], ['a', 'n'], ['a', 'p', 'p', 'l', 'e'], ['.']]]

                for xij in xi:
                    for xijk in xij:
                        word_counter[xijk] += 1
                        for xijkl in xijk:
                            char_counter[xijkl] += 1

    word_embed_mat, word2idx_dict = get_word2vec(glove_path, word_counter, limit)
    char2idx_dict = char2idx(char_counter, limit)
    return word_embed_mat, word2idx_dict, char2idx_dict


def process_file(glove_path, source_dir, target_dir, limit=-1, first_run=False):
    sent_tokenize = nltk.sent_tokenize
    if first_run is True:
        word_embed_mat, word2idx_dict, char2idx_dict = get_word_char_info(glove_path, source_dir, ['url', 'tweets'], -1)
        np.save(target_dir + os.sep + 'word_embed_mat.npy', np.array(word_embed_mat))
        json.dump(word2idx_dict, open(target_dir + os.sep + 'word2idx.json', 'w'))
        json.dump(char2idx_dict, open(target_dir + os.sep + 'char2idx.json', 'w'))

    with open(target_dir + os.sep + 'word2idx.json') as f:
        word2idx_dict = json.loads(f.read())
    with open(target_dir + os.sep + 'char2idx.json') as f:
        char2idx_dict = json.loads(f.read())
    print (len(word2idx_dict), len(char2idx_dict))
    # TWEETS: source_data = {data: [{uid:xxxx, name:xxxx, content:[{text:xxxx}, {text:xxxx},...,{text:xxxx}]},{},...,{}], data_type: tweets}
    # URLS: source_data = {data: [{uid:xxxx, url:xxxxx, content:[{text:xxxx}]},{},...,{}], data_type: urls}   
    for data_type in ['url', 'tweets']:
        print('Processing {} to word char ids...'.format(data_type))
        source_path = os.path.join(source_dir, 'mapped_{}_content.txt'.format(data_type))
        with open(source_path) as f:
            total_lines = len(f.readlines())
        source_data = open(source_path, 'r')

        for line in source_data: #tqdm(source_data, total=total_lines):
            raw_data = json.loads(line)
            data = raw_data['data']
            assert len(data.keys()) == 3
            #if data_type == 'url':
            #    _id = data['url_id']
            #else:
            _id = data['uid']

            if data['content'] is None:
                print(data_type, _id, 'has no content')
                data['content'] = [{'text': ''}]
                
            if data_type == 'url':
                assert len(data['content']) == 1
            out_dict = {'data_type': data_type, 'id': _id, 'content': []}
            for cont in data['content']:
                context = cont['text']
                context = context.encode('ascii', 'ignore').decode('utf-8')
                #words
                context = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', context)
                context = re.sub(r'[a-z]*[:.]+\S+', '', context)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')
                context = remove_emoji(context)

                #print(context)
                # eg. 'I have an apple.'
                xi = list(map(word_tokenize, sent_tokenize(context)))
                xi = [process_tokens(tokens) for tokens in xi] # [['I', 'have', 'an', 'apple', '.']]
                cxi = [[list(xijk) for xijk in xij] for xij in xi] # [[['I'], ['h', 'a', 'v', 'e'], ['a', 'n'], ['a', 'p', 'p', 'l', 'e'], ['.']]]

                one_content_dict = {'word': [], 'char': []}
                for xij in xi:
                    for xijk in xij:
                        one_content_dict['word'].append(word2idx_dict.get(xijk, 0))
                        char_list = []
                        for xijkl in xijk:
                            char_list.append(char2idx_dict.get(xijkl, 0))
                        one_content_dict['char'].append(char_list)
                out_dict['content'].append(one_content_dict)

            with open(target_dir + os.sep + 'encode_'+ data_type + os.sep + 'ids_%d.json' % (_id), 'w') as f:
                f.write(json.dumps(out_dict) + '\n')


def get_tf_idf_data(target_dir):
    source_path = '/home/dyou/url_recsys/data/url_tweet_tfidf_181201/'
    file_names = os.listdir(source_path)
    for file_name in file_names:
        file_path = source_path + file_name
        f = open(file_path, 'r')
        line = f.readline()
        out_dict = {}
        while line:
            data = json.loads(line)
            uid = data['uid_no']
            tweets = data['tweets']
            out_dict[int(uid)] = tweets
            line = f.readline()
        with open(target_dir + os.sep + 'tfidf200/' + file_name, 'w') as f:
            f.write(json.dumps(out_dict) + '\n')


if __name__ == '__main__':
    source_dir = '/home/dyou/url_recsys/data'  # mapped_url_content.txt, mapped_tweet_content.txt
    target_dir = '/home/dyou/url_recsys/dl_data/'
    glove_path = '/home/dyou/data/glove/glove.840B.300d.txt'
    process_file(glove_path, source_dir, target_dir, -1, False)
    #print('process tfidf......')
    #get_tf_idf_data(target_dir)



