import os
import re
import sys
from collections import OrderedDict
import codecs
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
import tqdm
from torch import autograd
from torch.autograd import Variable
import urllib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 80
plt.style.use('seaborn-pastel')
import time
embedding_path = 'data/glove.6B.100d.txt'
word_embedding_dim = 100
mapping_file = 'data/mapping_file.txt'
START_TAG = '<START>'
STOP_TAG = '<STOP>'
dropout = 0.5
hidden_size = 200
use_gpu = torch.cuda.is_available()
USE_CRF = True
reload_path = "./models/pre-trained-model"
new_type_path = '/media/sdd1/luosx/HuaiyuanYing/DecodeNER/all_other_experiments/datasets'

type_dataset = input("输入要得到的数据集:(NCBI,NCBI_ours,BC5-autoNER,BC5_ours")
if type_dataset == "NCBI":
    new_type_path = new_type_path+'/NCBI/'
if type_dataset == 'BC5-autoNER':
    new_type_path = new_type_path + '/BC5-autoNER/'
if type_dataset == 'BC5_ours':
    new_type_path = new_type_path+'/BC5-ours/'
if type_dataset == 'NCBI_ours':
    new_type_path = new_type_path + '/NCBI-ours/'
def input_data(path):
    insts = []
    seqs = []
    single_seq = []
    with open(path, 'r',encoding = 'utf-8') as file:
        for line in file.readlines():
            line = line.strip('\n')
            line = re.sub('\d','0',line)
            if line == '-DOCSTART- -X- -X- O':
                continue
            else:
                if line == '':
                    if len(seqs)!= 0:
                        insts.append(seqs)
                    seqs = []
                else:#这里都是一个样的
                    line = line.split(' ')
                    for word in line:
                        single_seq.append(word)
                    seqs.append(single_seq)
                    single_seq = []
    return insts

flag = []
def input_data_json(path_json):
    insts = []
    seqs = []
    single_seq = []
    one_line = []
    count = 0
    with open(path_json, 'r',encoding = 'utf-8') as file:
        line = file.read()
        line = line.split(']}')
        for inst in line[:-1]:
            single_seq = []
            inst = inst+']'
            inst = inst.split('],')
            str_words = inst[0]
            tags_ = inst[1]
            seqs = []
            one_line = []


            if count == 1:
                str_words = str_words.replace('str_words','')
                count = 0
            else:
                str_words = str_words.replace('str_words','')
                str_words = str_words.replace('"":','')
                str_words = str_words.replace('{','')
                str_words = str_words.replace(', ', '')
                str_words = str_words.replace('[', '')
                str_words = str_words.replace(']', '')
                str_words = str_words.replace(' ','',1)
                str_words = str_words.replace('CLS', '[CLS]')
            flag_1 = tags_.find('tags')
            flag.append(flag_1)
            tags_ = tags_.replace('tags','')
            tags_ = tags_.replace('"":', '')
            tags_ = tags_.replace(' [', '')
            tags_ = tags_.replace(']', '')
            tags_ = tags_.replace(' ', '', 1)
            tags_ = tags_.replace(' ','')
            tags_ = tags_.replace('O','0')
            tags_ = tags_.replace('B','1')
            tags_ = tags_.replace('I','2')
            tags_ = tags_.replace('"', '')
            str_words = str_words.strip('"')
            str_words = str_words.split('""')
            # str_words = str_words.strip('"')
            # str_words = str_words.split('""')

            single_seq.append(str_words)
            tags_ = tags_.split(',')

            single_seq.append(tags_)
            for single_str,single_tag in zip(str_words,tags_):
                seqs = []
                seqs.append(single_str)
                seqs.append(single_tag)
                one_line.append(seqs)
            insts.append(one_line)
    return insts
train_path = new_type_path +"train.json"
dev_path = new_type_path +"dev.json"
test_path = new_type_path +"test.json"
if type_dataset == 'BC5-autoNER':
    train_path = new_type_path + "train.json"
    dev_path = new_type_path+'dev.json'
    test_path = new_type_path+'test.json'
if type_dataset == 'BC5_ours':
    train_path = new_type_path + "train.json"
    dev_path = new_type_path + 'dev.json'
    test_path = new_type_path + 'test.json'

if type_dataset == 'NCBI_ours':
    train_path = new_type_path + "train.json"
    dev_path = new_type_path + 'dev.json'
    test_path = new_type_path + 'test.json'
if type_dataset == "NCBI":
    train_path = new_type_path + "train.json"
    dev_path = new_type_path + 'dev.json'
    test_path = new_type_path + 'test.json'

train_json = input_data_json(train_path)


dev_json = input_data_json(dev_path)
test_json = input_data_json(test_path)

def change_scheme(tags):
    """
    convert BIO->BIOES
    """
    new_tags = []
    for i,tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and tags[i+1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i+1 <len(tags) and tags[i+1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-','E-'))
    return new_tags

def change_whole_tags(seqs):
    for idx,seq in enumerate(seqs):
        tags = [w[-1] for w in seq]
        new_tags = change_scheme(tags)
        for word,new_tag in zip(seq,new_tags):
            word[-1] = new_tag

def remove_other_tags(insts):
    new_insts = []
    new_single_word_tag = []
    new_inst = []
    for inst in insts:
        for single_word_tag in inst:
            word = single_word_tag[0]
            tag = single_word_tag[-1]
            new_single_word_tag = [word,tag]
            new_inst.append(new_single_word_tag)
        new_insts.append(new_inst)
        new_inst = []

    return new_insts



def create_dict_two_list(instss):
    assert type(instss) is list
    dict = {}
    for insts in instss:
        for inst in insts:
            if inst not in dict:
                dict[inst] = 1
            else:
                dict[inst] += 1
    return dict

def create_dict_three_list(instss):
    assert type(instss) is list
    dict = {}
    for insts in instss:
        for inst in insts:
            for single_char in inst:
                if single_char not in dict:
                    dict[single_char] = 1
                else:
                    dict[single_char]+=1
    return dict


def creat_mapping(dict):
    sorted_items = sorted(dict.items(),key = lambda x: (-x[1] ,x[0]))
    idx_to_item = {idx: vocab[0] for idx,vocab in enumerate(sorted_items)}
    item_to_idx = {vocab: k for k, vocab in idx_to_item.items()}
    return item_to_idx, idx_to_item

def word_mapping(insts,lower):
    total_words = [[single_word[0].lower() if lower else single_word[0] for single_word in inst] for inst in insts]
    dict_word = create_dict_two_list(total_words)
    dict_word['<UNK>'] = 10000000
    word_to_idx, idx_to_word = creat_mapping(dict_word)
    print("Found %i unique words (%i in total)" % (
        len(dict_word), sum(len(x) for x in total_words)
    ))
    return dict_word,word_to_idx,idx_to_word
def char_mapping(instss):
    chars = [[[single_char for single_char in inst[0]]for inst in insts]for insts in instss]
    dict_char= create_dict_three_list(chars)
    char_to_idx,idx_to_char = creat_mapping(dict_char)
    print("Found %i unique characters" % len(dict_char))
    return dict_char, char_to_idx, idx_to_char

def tag_mapping(instss):
    tags =[[inst[-1] for inst in insts]for insts in instss]
    dict_tags = create_dict_two_list(tags)
    dict_tags[START_TAG] = -1
    dict_tags[STOP_TAG] = -2
    tag_to_idx,idx_to_tag = creat_mapping(dict_tags)
    print(dict_tags)
    print("Found %i unique named entity tags" % len(dict_tags))
    return dict_tags,tag_to_idx,idx_to_tag



dict_tags,tag_to_idx,idx_to_tag = tag_mapping(train_json)
dict_words,word_to_idx,idx_to_word = word_mapping(train_json,lower = True)
dict_chars,char_to_idx,idx_to_char = char_mapping(train_json)


def prepare_dataset(instss,word_to_idx,idx_to_word,tag_to_idx,idx_to_tag,char_to_idx,idx_to_char):
    dataset = []
    for insts in instss:
        str_words = [inst[0] for inst in insts]
        chars = [[char_to_idx[char] for char in word if char in char_to_idx]for word in str_words]
        tags = [tag_to_idx[inst[-1]]for inst in insts]
        words = [word_to_idx[word.lower() if word.lower() in word_to_idx else '<UNK>']for word in str_words]
        dataset.append({
            'str_words':str_words,
            'words': words,
            'chars':chars,
            'tags':tags
        })
    return dataset


train_dataset = prepare_dataset(train_json,word_to_idx,idx_to_word,tag_to_idx,idx_to_tag,char_to_idx,idx_to_char)
dev_dataset = prepare_dataset(dev_json,word_to_idx,idx_to_word,tag_to_idx,idx_to_tag,char_to_idx,idx_to_char)
test_dataset = prepare_dataset(test_json,word_to_idx,idx_to_word,tag_to_idx,idx_to_tag,char_to_idx,idx_to_char)


#load embedding
all_word_embeds = {}
with open(embedding_path,'r',encoding='utf-8') as file_embedding:
    line = file_embedding.readlines()
    count = 0
    for word in line:
        word = word.split()
        all_word_embeds[word[0]] = np.array([float(i) for i in word[1:]])



#initial word_embedding
word_embeddings = np.random.uniform(-np.sqrt(0.06),np.sqrt(0.06),(len(word_to_idx),word_embedding_dim))

for word in word_to_idx:
    if word in all_word_embeds:
        word_embeddings[word_to_idx[word]] = all_word_embeds[word]
    else:
        if word.lower() in all_word_embeds:
            word_embeddings[word_to_idx[word.lower()]] = all_word_embeds[word.lower()]

print('the length of load embedding is {}'.format(len(word_embeddings)))



#some functions
def log_sum_exp(vec):
    '''
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * tagset_size
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    _, idx = torch.max(vec,1)
    return to_scalar(idx)

def to_scalar(var):
    return var.view(-1).data.tolist()[0]


#CALCULATE SCORE
def score_sentences(self, feats, tags):
    # tags is ground_truth, a list of ints, length is len(sentence)
    # feats is a 2D tensor, len(sentence) * tagset_size
    r = torch.LongTensor(range(feats.size()[0]))
    if self.use_gpu:
        r = r.cuda()
        pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_idx[START_TAG]]), tags])
        pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_idx[STOP_TAG]])])
    else:
        pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_idx[START_TAG]]), tags])
        pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_idx[STOP_TAG]])])

    score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

    return score


def forward_alg(self, feats):
    '''
    This function performs the forward algorithm explained above
    '''
    # calculate in log domain
    # feats is len(sentence) * tagset_size
    # initialize alpha with a Tensor with values all equal to -10000.

    # Do the forward algorithm to compute the partition function
    init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)

    # START_TAG has all of the score.
    init_alphas[0][self.tag_to_idx[START_TAG]] = 0.

    # Wrap in a variable so that we will get automatic backprop
    forward_var = autograd.Variable(init_alphas)
    if self.use_gpu:
        forward_var = forward_var.cuda()

    # Iterate through the sentence
    for feat in feats:
        # broadcast the emission score: it is the same regardless of
        # the previous tag
        emit_score = feat.view(-1, 1)

        # the ith entry of trans_score is the score of transitioning to
        # next_tag from i
        tag_var = forward_var + self.transitions + emit_score

        # The ith entry of next_tag_var is the value for the
        # edge (i -> next_tag) before we do log-sum-exp
        max_tag_var, _ = torch.max(tag_var, dim=1)

        # The forward variable for this tag is log-sum-exp of all the
        # scores.
        tag_var = tag_var - max_tag_var.view(-1, 1)

        # Compute log sum exp in a numerically stable way for the forward algorithm
        forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)  # ).view(1, -1)
    terminal_var = (forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]).view(1, -1)
    alpha = log_sum_exp(terminal_var)
    # Z(x)
    return alpha


def viterbi_algo(self, feats):
    '''
    In this function, we implement the viterbi algorithm explained above.
    A Dynamic programming based approach to find the best tag sequence
    '''
    backpointers = []
    # analogous to forward

    # Initialize the viterbi variables in log space
    init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
    init_vvars[0][self.tag_to_idx[START_TAG]] = 0

    # forward_var at step i holds the viterbi variables for step i-1
    forward_var = Variable(init_vvars)
    if self.use_gpu:
        forward_var = forward_var.cuda()
    for feat in feats:
        next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
        _, bptrs_t = torch.max(next_tag_var, dim=1)
        bptrs_t = bptrs_t.squeeze().data.cpu().numpy()  # holds the backpointers for this step
        next_tag_var = next_tag_var.data.cpu().numpy()
        viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]  # holds the viterbi variables for this step
        viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
        if self.use_gpu:
            viterbivars_t = viterbivars_t.cuda()

        # Now add in the emission scores, and assign forward_var to the set
        # of viterbi variables we just computed
        forward_var = viterbivars_t + feat
        backpointers.append(bptrs_t)

    # Transition to STOP_TAG
    terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
    terminal_var.data[self.tag_to_idx[STOP_TAG]] = -10000.
    terminal_var.data[self.tag_to_idx[START_TAG]] = -10000.
    best_tag_id = argmax(terminal_var.unsqueeze(0))
    path_score = terminal_var[best_tag_id]

    # Follow the back pointers to decode the best path.
    best_path = [best_tag_id]
    for bptrs_t in reversed(backpointers):
        best_tag_id = bptrs_t[best_tag_id]
        best_path.append(best_tag_id)

    # Pop off the start tag (we dont want to return that to the caller)
    start = best_path.pop()
    assert start == self.tag_to_idx[START_TAG]  # Sanity check
    best_path.reverse()
    return path_score, best_path


def forward_calc(self, sentence, chars, chars2_length, d):
    '''
    The function calls viterbi decode and generates the
    most probable sequence of tags for the sentence
    '''

    # Get the emission scores from the BiLSTM
    feats = self._get_lstm_features(sentence, chars, chars2_length, d)
    # viterbi to get tag_seq

    # Find the best path, given the features.
    if self.use_crf:
        score, tag_seq = self.viterbi_decode(feats)
    else:
        score, tag_seq = torch.max(feats, 1)
        tag_seq = list(tag_seq.cpu().data)

    return score, tag_seq




#initial weight

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)

def init_lstm(input_lstm):
    for ind in range(0,input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        sample_range = np.sqrt(6.0/(weight.size(0)+weight.size(1)))
        nn.init.uniform_(weight,-sample_range,sample_range)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        sample_range = np.sqrt(6.0 / (weight.size(0) + weight.size(1)))
        nn.init.uniform_(weight,-sample_range,+sample_range)

    if input_lstm.bidirectional:
        for ind in range(0,input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l'+str(ind)+'_reverse')
            sample_range = np.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            nn.init.uniform_(weight, -sample_range, +sample_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sample_range = np.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            nn.init.uniform_(weight, -sample_range, +sample_range)
    if input_lstm.bias:
        for ind in range(0,input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l'+str(ind))
            bias.data.zero_()
            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size:2*input_lstm.hidden_size] = 1
            bias = eval('input_lstm.bias_hh_l'+str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_linear(input_linear):
    bias = np.sqrt(6.0/(input_linear.weight.size(0)+input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight,-bias,bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def get_lstm_features(self, sentence, chars2, chars2_length, d):
    if self.char_mode == 'LSTM':

        chars_embeds = self.char_embeds(chars2).transpose(0, 1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)

        lstm_out, _ = self.char_lstm(packed)

        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)

        outputs = outputs.transpose(0, 1)

        chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))

        if self.use_gpu:
            chars_embeds_temp = chars_embeds_temp.cuda()

        for i, index in enumerate(output_lengths):
            chars_embeds_temp[i] = torch.cat(
                (outputs[i, index - 1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))

        chars_embeds = chars_embeds_temp.clone()

        for i in range(chars_embeds.size(0)):
            chars_embeds[d[i]] = chars_embeds_temp[i]

    if self.char_mode == 'CNN':
        chars_embeds = self.char_embeds(chars2).unsqueeze(1)

        ## Creating Character level representation using Convolutional Neural Netowrk
        ## followed by a Maxpooling Layer
        chars_cnn_out3 = self.char_cnn3(chars_embeds)
        chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0),
                                                                                              self.out_channels)

        ## Loading word embeddings
    embeds = self.word_embeds(sentence)

    ## We concatenate the word embeddings and the character level representation
    ## to create unified representation for each word
    embeds = torch.cat((embeds, chars_embeds), 1)

    embeds = embeds.unsqueeze(1)

    ## Dropout on the unified embeddings
    embeds = self.dropout(embeds)

    ## Word lstm
    ## Takes words as input and generates a output at each step
    lstm_out, _ = self.lstm(embeds)

    ## Reshaping the outputs from the lstm layer
    lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)

    ## Dropout on the lstm output
    lstm_out = self.dropout(lstm_out)

    ## Linear layer converts the ouput vectors to tag space
    lstm_feats = self.hidden2tag(lstm_out)

    return lstm_feats

def get_neg_log_likelihood(self, sentence, tags, chars2, chars2_length, d):
    # sentence, tags is a list of ints
    # features is a 2D tensor, len(sentence) * self.tagset_size
    feats = self._get_lstm_features(sentence, chars2, chars2_length, d)

    if self.use_crf:
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
    else:
        tags = Variable(tags)
        scores = nn.functional.cross_entropy(feats, tags)
        return scores

class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size,tag_to_idx,embedding_dim,hidden_size,
                 char_to_idx = None,pre_word_embeds = None,char_out_dim = 30,
                 char_embedding_dim = 30,use_gpu = False,use_crf = True,
                 char_mode = 'CNN'
    ):
        '''
        Input parameters:

                vocab_size= Size of vocabulary (int)
                tag_to_ix = Dictionary that maps NER tags to indices
                embedding_dim = Dimension of word embeddings (int)
                hidden_dim = The hidden dimension of the LSTM layer (int)
                char_to_ix = Dictionary that maps characters to indices
                pre_word_embeds = Numpy array which provides mapping from word embeddings to word indices
                char_out_dimension = Output dimension from the CNN encoder for character
                char_embedding_dim = Dimension of the character embeddings
                use_gpu = defines availability of GPU,
                    when True: CUDA function calls are made
                    else: Normal CPU function calls are made
                use_crf = parameter which decides if you want to use the CRF layer for output decoding
        '''
        super(BiLSTM_CRF,self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_size
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_idx)
        self.out_channels = char_out_dim
        self.char_mode = char_mode

        #embedding layer
        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim
            self.char_embeds = nn.Embedding(len(char_to_idx),char_embedding_dim)
            init_embedding(self.char_embeds.weight)
            self.char_cnn3 = nn.Conv2d(in_channels=1,out_channels=self.out_channels,kernel_size=(3,char_embedding_dim),padding=(1,0))
        self.word_embeds = nn.Embedding(vocab_size,embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(dropout)

        #lstm layer
        self.lstm = nn.LSTM(embedding_dim+self.out_channels,hidden_size,bidirectional=True)
        init_lstm(self.lstm)

        #hidden to tag
        self.hidden2tag = nn.Linear(hidden_size*2,self.tagset_size)
        init_linear(self.hidden2tag)

        if self.use_crf:
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size)
            )
            self.transitions.data[tag_to_idx[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

    _score_sentence = score_sentences
    _get_lstm_features = get_lstm_features
    _forward_alg = forward_alg
    viterbi_decode = viterbi_algo
    neg_log_likelihood = get_neg_log_likelihood
    forward = forward_calc

#creating the model using the Class defined above
model = BiLSTM_CRF(vocab_size=len(word_to_idx),
                   tag_to_idx=tag_to_idx,
                   embedding_dim=word_embedding_dim,
                   hidden_size=hidden_size,
                   use_gpu=use_gpu,
                   char_to_idx=char_to_idx,
                   pre_word_embeds=word_embeddings)

print("Model Initialized!!!")


if use_gpu:
    model.cuda()




learning_rate = 0.015
momentum = 0.9
num_of_epoch = 50
decay_rate = 0.05
gradient_clip = 5.0
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate,momentum = momentum)

losses = []
loss = 0.0

best_dev_F = -1.0 # Current best F-1 Score on Dev Set
best_test_F = -1.0 # Current best F-1 Score on Test Set
best_train_F = -1.0 # Current best F-1 Score on Train Set
all_F = [[0, 0, 0]] # List storing all the F-1 Scores
eval_every = len(train_dataset) # Calculate F-1 Score after this many iterations
plot_every = 2000 # Store loss after this many iterations
count = 0 #Counts the number of iterations


def evaluating(model, datas, best_F, dataset="Train"):
    '''
    The function takes as input the model, data and calcuates F-1 Score
    It performs conditional updates
     1) Flag to save the model
     2) Best F-1 score
    ,if the F-1 score calculated improves on the previous F-1 score
    '''
    # Initializations
    prediction = []  # A list that stores predicted tags
    save = False  # Flag that tells us if the model needs to be saved
    new_F = 0.0  # Variable to store the current F1-Score (may not be the best)
    correct_preds = []
    total_correct = []
    total_preds = []  # Count variables
    path = 'data_' + type_dataset
    if not os.path.exists(path):
        os.makedirs(path)
    output_txt_path = path+'/'+type_dataset+'_original_pred.txt'
    if os.path.exists(output_txt_path):
        os.remove(output_txt_path)

    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']#len(seq) * num_of_char_for_a_single_word *char_vocab_size


        if True:
            d = {}

            # Padding the each word to max word size of that sentence
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))#len(seq) * char_max_len *char_size

        dwords = Variable(torch.LongTensor(data['words']))#len(seq)*vocab_size

        # We are getting the predicted output from our model
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, chars2_length, d)
        predicted_id = out
        y_pred = change_ner(predicted_id)


        y_true = change_ner(ground_truth_id)
        # with open('pred_true.txt',"a+",encoding="UTF-8") as testfile:
        #     testfile.write(str(y_pred))
        #     testfile.write('\n')
        #     testfile.write(str(predicted_id))
        #     testfile.write('\n')
        #     testfile.write(str(y_true))
        #     testfile.write('\n')
        #     testfile.write(str(ground_truth_id))
        #     testfile.write('\n')
        #     testfile.write('\n')
        #     testfile.close()
        total_correct.append(y_true)
        total_preds.append(y_pred)
        if dataset == 'Test':
            with open(output_txt_path, 'a+', encoding="UTF-8") as file:
                file.write(str(words))
                file.write('    ')
                file.write(str(y_pred))
                file.write('\n')
                file.close()

    new_pred = []
    new_true = []

    for idx,inst in enumerate(total_correct):
        single_pred = []
        single_true = []
        if inst == 'A':
            continue
        else:
            single_pred = total_preds[idx]
            single_true = total_correct[idx]
            new_pred.append(single_pred)
            new_true.append(single_true)
    print("accuary: ", accuracy_score(total_correct, total_preds))
    print("p: ", precision_score(total_correct, total_preds))
    print("r: ", recall_score(total_correct, total_preds))
    print("f1: ", f1_score(total_correct, total_preds))
    print("classification report: ")
    print(classification_report(total_correct, total_preds))
    new_F = f1_score(total_correct, total_preds)
    with open(path+"/outcome_original.txt", "a+", encoding="UTF-8") as file:
        file.write("\n")
        file.write("dataset:" + dataset)
        file.write("\n")
        file.write("accuracy：")
        file.write("\n")
        file.write(str(accuracy_score(total_correct, total_preds)))
        file.write("\n")
        file.write("precision_score:")
        file.write("\n")
        file.write(str(precision_score(total_correct, total_preds)))
        file.write("\n")
        file.write("recall_score:")
        file.write("\n")
        file.write(str(recall_score(total_correct, total_preds)))
        file.write("\n")
        file.write("f1_score")
        file.write(str(f1_score(total_correct, total_preds)))
        file.write("\n")
        file.write(classification_report(total_correct, total_preds))
        file.write("\n")
    new_F = f1_score(new_true, new_pred)
    # If our current F1-Score is better than the previous best, we update the best
    # to current F1 and we set the flag to indicate that we need to checkpoint this model

    if new_F > best_F:
        best_F = new_F
        save = True

    return best_F, new_F, save

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def change_ner(y):
    new_y = []
    singer_y = 'O'
    for inst in y:
        if type(inst) != int:
            inst = inst.item()
        inst = idx_to_tag[inst]
        singer_y = 'O'
        if inst == '0':
            singer_y = 'O'
        elif inst == '1':
            singer_y = 'B-Disease'
        elif inst == '2':
            singer_y = 'I-Disease'
        elif inst == '3':
            singer_y = 'B-Chemical'
        elif inst == '4':
            singer_y = 'I-Chemical'
        else:
            singer_y = 'A'
        new_y.append(singer_y)
    return new_y
best_dev__F = 0
if True:
    tr = time.time()
    model.train(True)
    for epoch in trange(1, 50):
        for i, index in enumerate(np.random.permutation(len(train_dataset))):
            count += 1
            data = train_dataset[index]

            ##gradient updates for each data entry
            model.zero_grad()

            sentence_in = data['words']
            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']
            chars2 = data['chars']#len(seq) * num_of_char_for_a_single_word *char_vocab_size


            if True:

                d = {}

                ## Padding the each word to max word size of that sentence
                chars2_length = [len(c) for c in chars2]#c is a word,of course the word is len(word)*char_vocab_size
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')#len(seq)*num_of_char_for_a_single_word
                for i, c in enumerate(chars2):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))


            targets = torch.LongTensor(tags)

            # we calculate the negative log-likelihood for the predicted tags using the predefined function
            if use_gpu:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(),
                                                              chars2_length, d)
            else:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length, d)


            loss += neg_log_likelihood.item() / len(data['words'])
            neg_log_likelihood.backward()

            # we use gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            # Storing loss
            if count % plot_every == 0:
                loss /= plot_every
                print(count, ': ', loss)
                if losses == []:
                    losses.append(loss)
                losses.append(loss)
                loss = 0.0

            # Evaluating on Train, Test, Dev Sets
            if count%10000==0:
                model.train(False)
                #best_train_F, new_train_F, _ = evaluating(model, train_dataset, best_train_F, "Train")
                best_dev_F, new_dev_F, save = evaluating(model, dev_dataset, best_dev_F, "Dev")
                # model_name = "./models/trained"
                # if save:
                #     print("Saving Model to ", model_name)
                #     torch.save(model.state_dict(), model_name)
                path = 'data_' + type_dataset
                if new_dev_F >best_dev__F:
                    torch.save(model.state_dict(), path+"/"+type_dataset+"_original.pth")
                    best_dev__F = new_dev_F
                model.train(True)

            # Performing decay on the learning rate
            if count % len(train_dataset) == 0:
                adjust_learning_rate(optimizer, lr=learning_rate / (1 + decay_rate * count / len(train_dataset)))
path = 'data_' + type_dataset
model.load_state_dict(torch.load(path+"/"+type_dataset+"_original.pth"))
model.train(False)
best_test_F, new_test_F, _ = evaluating(model, test_dataset, best_test_F, "Test")
print("best_test_F = ",best_test_F)
print("new_test_F = ", new_test_F)