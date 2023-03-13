import json
import os
import math
import random

import torch
import heapq
# import random
import pickle
import datetime
from rouge import rouge
from bleu import compute_bleu
from tqdm import tqdm
import time
from random import *
import copy
# from sentence_transformers import SentenceTransformer, util
# model_sentence = SentenceTransformer('all-MiniLM-L6-v2')

def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class WordDictionary:
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>','<mask>','<cls>']
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.__word2count = {}

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}





class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader:
    def __init__(self, data_path, index_dir, vocab_size, args):
        self.word_dict = WordDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()


        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.item_keyword = args.item_keyword
        self.item_keyword_path = args.item_keyword_path
        self.args = args
        self.image_fea_path = args.image_fea_path
        self.feature_extract_path = args.feature_extract_path

        self.initialize(data_path)
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        self.__bos = self.word_dict.word2idx['<bos>']
        self.__eos = self.word_dict.word2idx['<eos>']
        self.__pad = self.word_dict.word2idx['<pad>']
        self.__mask = self.word_dict.word2idx['<mask>']
        self.__cls = self.word_dict.word2idx['<cls>']
        self.user_profile = json.load(open(args.user_profile_path,'rb'))
        self.item_profile = json.load(open(args.item_profile_path,'rb'))

        self.reviews, self.reviews_neg= self.prep_profile_data_list(smax=self.args.profile_words, dmax=self.args.profile_len, data_path = args.data_path)
        # self.user_profile = self.prep_user_profile_data_list(self.user_profile,self.args.words, args.profile_len)
        # self.item_profile = self.prep_item_profile_data_list(self.item_profile,self.args.words, args.profile_len)
        self.max_mask = 3
        self.feature_set = set()
        self.item_keyword_set = set()
        self.train, self.valid, self.test = self.load_data(args,self.image_fea_path, data_path, index_dir)


        # self.train, self.valid, self.test = self.load_data(self.image_fea_path, data_path, index_dir)


    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        if self.item_keyword:
            self.item_keywords_dict = pickle.load(open(self.item_keyword_path, 'rb'))
        if self.args.feature_extract:
            feature_extra = pickle.load(open(self.feature_extract_path, 'rb'))
        cnt = 0
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            (fea, adj, tem, sco) = review['template']
            self.word_dict.add_sentence(tem)
            if self.args.feature_extract:
                fea = feature_extra[cnt]
                cnt += 1

            self.word_dict.add_word(fea)

            if self.item_keyword:
                keyword = self.item_keywords_dict[review['item']]
                self.word_dict.add_sentence(keyword)
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def pad_to_max(self, seq, seq_max,pad_token = 0):

        while (len(seq) < (seq_max+2)):
            seq.append(pad_token)
        return seq[:seq_max+2]

    def pad_to_max_list(self, seq, seq_max,pad_token = 0):

        while (len(seq) < (seq_max)):
            seq.append(pad_token)
        return seq[:seq_max]

    def sentence_index(self,sentences,smax,dmax):
        loop = range(0, len(sentences))
        new_data = []
        for idx in loop:


            data_list = [self.__bos] + sentences[idx][:smax] + [self.__eos]
            sent_lens = len(data_list)
            if sent_lens == 0:
                continue
            if sent_lens > smax:
                sent_lens = smax

            _data_list = self.pad_to_max(data_list, smax, pad_token=self.__pad)
            new_data.append(_data_list)
            # data_lengths.append(sent_lens)
            if len(new_data) >= dmax:  # skip if already reach dmax
                break
        new_data = self.pad_to_max_list(new_data, dmax,pad_token=[self.__bos] + [self.__eos] + [self.__pad for _ in range(smax)])
        return new_data



    def prep_profile_data_list(self, smax, dmax, data_path):
        import random
        def negative_sample(index, review):
            max_index = len(review) - 1
            if index == 0:
                negative = random.choice(review[1:])
            elif index == max_index:
                negative = random.choice(review[:-1])
            else:
                tmp1 = review[0:index]
                tmp2 = review[index + 1:]
                tmp1.extend(tmp2)
                negative = random.choice(tmp1)
            return negative
        reviews = pickle.load(open(data_path, 'rb'))
        review_negs= []
        root = os.path.split(data_path)[0]
        review_negs_path = os.path.join(root,'reviews_neg.pickle')
        review_poss_path = os.path.join(root,'reviews_pos.pickle')

        # if os.path.exists(review_negs_path):
        #
        #     review_negs = pickle.load(open(review_negs_path,'rb'))
        #     for index, review in tqdm(enumerate(reviews)):
        #         # mu, mi = review['user_item_profile_mask'][0], review['user_item_profile_mask'][1]
        #         # user_profile, item_profile = [], []
        #         # reviews_copy = copy.deepcopy(reviews)
        #         user_profile = self.user_profile[review['user']].copy()
        #         item_profile = self.item_profile[review['item']].copy()
        #         u_p = [user_profile[i] for i in review['user_profile.index'].tolist()]
        #         i_p = [item_profile[i] for i in review['item_profile.index'].tolist()]
        #         u_index = [self.seq2ids(seq) for seq in u_p]
        #         i_index = [self.seq2ids(seq) for seq in i_p]
        #         review['user_profile'] = self.sentence_index(u_index, smax, dmax)
        #         review['item_profile'] = self.sentence_index(i_index, smax, dmax)
        #         review['cls'] = 1
        if os.path.exists(review_negs_path) and os.path.exists(review_poss_path):

            review_negs = pickle.load(open(review_negs_path,'rb'))
            reviews = pickle.load(open(review_poss_path,'rb'))

            print('Directly use')
        else:
            for index, review in tqdm(enumerate(reviews)):
                # mu, mi = review['user_item_profile_mask'][0], review['user_item_profile_mask'][1]
                # user_profile, item_profile = [], []
                # reviews_copy = copy.deepcopy(reviews)
                user_profile = self.user_profile[review['user']].copy()
                item_profile = self.item_profile[review['item']].copy()
                u_p = [user_profile[i] for i in review['user_profile.index'].tolist()]
                i_p = [item_profile[i] for i in review['item_profile.index'].tolist()]
                u_index = [self.seq2ids(seq) for seq in u_p]
                i_index = [self.seq2ids(seq) for seq in i_p]
                review['user_profile'] = self.sentence_index(u_index, smax, dmax)
                review['item_profile'] = self.sentence_index(i_index, smax, dmax)
                review_neg = copy.deepcopy(review)
                negative = negative_sample(index, reviews)
                negative = negative['template']
                review_neg['template'] = negative
                review_neg['cls'] = 0
                review['cls'] = 1
                review_negs.append(review_neg)
            pickle.dump(review_negs, open(review_negs_path,'wb'))



        return reviews, review_negs

    def prep_user_profile_data_list(self, raw_data, smax, dmax):

        data = dict()
        for key, value in raw_data.items():
            data[self.user_dict.entity2idx[key]] = [self.seq2ids(seq) for seq in self.user_profile[key]]

        all_data = dict()
        # all_lengths = dict()
        for key, value in data.items():
            new_data = []
            # data_lengths = []

            loop = range(0, len(value))
            for idx in loop:

                data_list = [self.__bos] + value[idx][:smax] + [self.__eos]
                sent_lens = len(data_list)
                if sent_lens == 0:
                    continue
                if sent_lens > smax:
                    sent_lens = smax

                _data_list = self.pad_to_max(data_list, smax, pad_token=self.__pad)
                new_data.append(_data_list)
                # data_lengths.append(sent_lens)
                if len(new_data) >= dmax:  # skip if already reach dmax
                    break
            new_data = self.pad_to_max_list(new_data, dmax,  # dmax - early skip!
                                            pad_token=[self.__bos] + [self.__eos] + [self.__pad for _ in range(smax)])

            # data_lengths = self.pad_to_max(data_lengths, dmax)

            all_data[key] = new_data
            # all_lengths[key] = data_lengths
        return all_data


    def prep_item_profile_data_list(self, raw_data, smax, dmax):

        data = dict()
        for key, value in raw_data.items():
            data[self.item_dict.entity2idx[key]] = [self.seq2ids(seq) for seq in self.item_profile[key]]

        all_data = dict()
        # all_lengths = dict()
        for key, value in data.items():
            new_data = []
            data_lengths = []

            loop = range(0, len(value))
            for idx in loop:
                data_list = [self.__bos] + value[idx][:smax] + [self.__eos]
                sent_lens = len(data_list)
                if sent_lens == 0:
                    continue
                if sent_lens > smax:
                    sent_lens = smax

                _data_list = self.pad_to_max(data_list, smax, pad_token=self.__pad)
                new_data.append(_data_list)
                # data_lengths.append(sent_lens)
                if len(new_data) >= dmax:  # skip if already reach dmax
                    break
            new_data = self.pad_to_max_list(new_data, dmax,  # dmax - early skip!
                                       pad_token=[self.__bos] + [self.__eos] + [self.__pad for _ in range(smax)])

            # data_lengths = self.pad_to_max(data_lengths, dmax)

            all_data[key] = new_data
            # all_lengths[key] = data_lengths
        return all_data



    def load_data(self, args, dir, data_path, index_dir):

        def mask(text_):
            text = copy.deepcopy(text_)
            n_mask = min(self.max_mask, max(1, int(round(len(text) * 0.30))))
            cand_maked_pos = [i for i, token in enumerate(text)]
            shuffle(cand_maked_pos)

            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:n_mask]:  ## 取其中的三个；masked_pos=[6, 5, 17] 注意这里对应的是position信息；masked_tokens=[13, 9, 16] 注意这里是被mask的元素之前对应的原始单字数字；
                masked_pos.append(pos)
                masked_tokens.append(text[pos])
                if random() < 0.8:  # 80%
                    text[pos] = self.__mask  # make mask
                elif random() < 0.5:  # 10%
                    index = randint(0, len(self.word_dict) - 1)  # random index in vocabulary
                    text[pos] = index  # replace
            if self.max_mask > n_mask:
                n_pad = self.max_mask - n_mask
                masked_tokens.extend([self.__pad] * n_pad)  ##  masked_tokens= [13, 9, 16, 0, 0] masked_tokens 对应的是被mask的元素的原始真实标签是啥，也就是groundtruth
                masked_pos.extend([self.__pad] * n_pad)
            return text

        def limit_len(sentence, max_len):
            length = len(sentence)
            if length >= max_len:
                return sentence[:max_len]
            else:
                return sentence

        seq_len = args.words
        data = []
        data_neg = []
        reviews = self.reviews
        reviews_neg = self.reviews_neg
        # reviews = pickle.load(open(data_path, 'rb'))

        cnt = 0
        if args.feature_extract :
            feature_extra = pickle.load(open(self.feature_extract_path, 'rb'))


        for review in reviews:
            (fea, adj, tem, sco) = review['template']
            if self.args.feature_extract:
                fea = feature_extra[cnt]
                cnt += 1

            text_id = self.seq2ids(tem)
            text_id = limit_len(text_id, seq_len)
            text_id_masked = mask(text_id)
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'text': text_id,
                         'text_masked': text_id_masked,
                         'feature': self.word_dict.word2idx.get(fea, self.__unk),
                         'adjective': self.word_dict.word2idx.get(adj, self.__unk),
                         # 'item_keyword': self.seq2ids(self.item_keywords_dict[review['item']]),
                         'item_keyword': self.word_dict.word2idx['<unk>'],
                         # 'user_profile':user_profile,
                         # 'item_profile':item_profile})
                         'user_profile':review['user_profile'],
                         'item_profile':review['item_profile'],
                         'cls':review['cls']})
            if fea in self.word_dict.word2idx:
                self.feature_set.add(fea)
            else:
                self.feature_set.add('<unk>')

        for review in reviews_neg:
            (fea, adj, tem, sco) = review['template']
            if self.args.feature_extract:
                fea = feature_extra[cnt]
                cnt += 1

            text_id = self.seq2ids(tem)
            text_id = limit_len(text_id, seq_len)
            text_id_masked = mask(text_id)
            data_neg.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'text': text_id,
                         'text_masked': text_id_masked,
                         'feature': self.word_dict.word2idx.get(fea, self.__unk),
                         'adjective': self.word_dict.word2idx.get(adj, self.__unk),
                         'item_keyword': self.word_dict.word2idx['<unk>'],

                             # 'item_keyword': self.seq2ids(self.item_keywords_dict[review['item']]),
                         # 'user_profile':user_profile,
                         # 'item_profile':item_profile})

                         'user_profile': review['user_profile'],
                         'item_profile': review['item_profile'],
                         'cls': review['cls']})



        print('pos:{}'.format(len(data)))
        print('neg:{}'.format(len(data_neg)))
        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []


        for idx in train_index:
            train.append(data[idx])
            if random() < args.neg_rate:
                train.append(data_neg[idx])
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test


    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index


def sentence_format(sentence, max_len, pad, bos, eos):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)
def keyword_format(keyword, max_len, pad):
    length = len(keyword)
    if length >= max_len:
        return  keyword[:max_len]
    else:
        return keyword + [pad] * (max_len - length)



class Batchify:
    def __init__(self, data, word2idx, args, idx2item, seq_len=15, key_len=2, batch_size=128, shuffle=False):
        bos = word2idx['<bos>']
        eos = word2idx['<eos>']
        pad = word2idx['<pad>']
        mask = word2idx['<mask>']
        u, i, r, t, t_masked, f, a, i_k , u_p, i_p, cls = [], [], [], [],[], [], [], [], [], [], []

        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append(sentence_format(x['text'], seq_len, pad, bos, eos))
            t_masked.append(sentence_format(x['text_masked'], seq_len, pad, bos, eos))
            f.append([x['feature']])
            a.append([x['adjective']])
            # i_k.append(keyword_format(x['item_keyword'],key_len,pad))
            i_k.append(x['item_keyword'])

            u_p.append(x['user_profile'])
            i_p.append(x['item_profile'])
            cls.append(x['cls'])



        self.idx2item = idx2item
        self.image_fea_path=args.image_fea_path
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.seq_masked = torch.tensor(t_masked, dtype=torch.int64).contiguous()
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        self.adjective = torch.tensor(a, dtype=torch.int64).contiguous()
        self.args = args
        self.item_k = torch.tensor(i_k, dtype=torch.int64).contiguous()
        self.user_profile = torch.tensor(u_p, dtype=torch.int64).contiguous()
        self.item_profile = torch.tensor(i_p, dtype=torch.int64).contiguous()
        self.cls = torch.tensor(cls, dtype=torch.int64).contiguous()


        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def get_img(self,dir,names):
        fea= []
        for i in range(len(names)):
            img_name = names[i]
            path_img = os.path.join(dir, img_name)
            mean_img = os.path.join(dir, 'mean')
            if os.path.exists(path_img):
                fea.append(pickle.load(open(path_img, 'rb')))
            else:
                fea.append(pickle.load(open(mean_img, 'rb')))
        fea = torch.stack(fea)
        return fea

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        seq_masked = self.seq_masked[index]  # (batch_size, seq_len)
        feature = self.feature[index]  # (batch_size, 1)
        item_k = self.item_k[index]
        user_profile = self.user_profile[index]
        item_profile = self.item_profile[index]
        adjective = self.adjective[index]
        cls = self.cls[index]
        item_tmp=item.tolist()
        names = []
        for i in item_tmp:
            names.append(self.idx2item[i])
        if self.args.image_fea:
            image_fea = self.get_img(self.image_fea_path, names)
        else:
            image_fea = torch.tensor([[]])

        return user, item, rating, seq, seq_masked, feature, adjective, item_k, image_fea, user_profile, item_profile, cls



def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def ids2tokens(ids, word2idx, idx2word):
    eos = word2idx['<eos>']
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens
