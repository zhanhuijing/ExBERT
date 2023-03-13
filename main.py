import os
import math
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

# from module import PETER
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
from TRM import *
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
def str2bool(str):
    return True if str.lower() == 'true' else False

parser = argparse.ArgumentParser(description='Transformer')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                    help='load indexes')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of embeddings')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the transformer')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs_teacher', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--epochs_student', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./peter/',
                    help='directory to save the final model')
parser.add_argument('--outf_teacher', type=str, default='generated_teacher.txt',
                    help='output file for generated text')
parser.add_argument('--outf_student', type=str, default='generated_student.txt',
                    help='output file for generated text')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--rating_reg', type=float, default=0.1,
                    help='regularization on recommendation task')
parser.add_argument('--context_reg', type=float, default=1.0,
                    help='regularization on context prediction task')
parser.add_argument('--text_reg', type=float, default=1.0,
                    help='regularization on text generation task')
parser.add_argument('--img_reg', type=float, default=0.01,
                    help='regularization on text generation task')
parser.add_argument('--peter_mask', action='store_true',
                    help='True to use peter mask; Otherwise left-to-right mask')
parser.add_argument('--use_feature', action='store_true',
                    help='False: no feature; True: use the feature')
parser.add_argument('--words', type=int, default=15,
                    help='number of words to generate for each sample')
parser.add_argument('--profile_words', type=int, default=15,
                    help='number of words to generate for each sample')
parser.add_argument('--profile_len', type=int, default=5,
                    help='number of words to generate for each sample')
parser.add_argument('--key_len', type=int, default=2,
                    help='number of words to generate for each sample')

parser.add_argument('--test_step', type=int, default=1,
                    help='number of words to generate for each sample')
parser.add_argument('--T', type=float, default=2,
                    help='temperture for KD')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='KD weight')

parser.add_argument('--clsf_reg', type=float, default=0.5,
                    help='clsf_reg')
parser.add_argument('--neg_rate', type=float, default=0.5,
                    help='neg_rate')
parser.add_argument('--sim', action='store_true', help='sim_loss')
parser.add_argument('--mse', action='store_true', help='fea_loss')
parser.add_argument('--mae', action='store_true', help='fea_loss')
parser.add_argument('--eval_teacher_train', action='store_true', help='fea_loss')
parser.add_argument('--eva_teacher', action='store_true', help='user_item_profile')


parser.add_argument('--cosinelr', type=str2bool, default=False, help='cosine learning rate')
parser.add_argument('--scheduler_lr', type=str2bool, default=True, help='scheduler')
parser.add_argument('--feature_extract', action='store_true', help='feature_extract')
parser.add_argument('--feature_extract_path', type=str, default=None, help='path for feature_keybert')
parser.add_argument('--user_item_profile', action='store_true', help='user_item_profile')
parser.add_argument('--user_profile_path', type=str, default='D:/PETER2/Amazon/Movies_and_TV/user_profile.json', help='path for user_profile_path')
parser.add_argument('--item_profile_path', type=str, default='D:/PETER2/Amazon/Movies_and_TV/item_profile.json', help='path for item_profile_path')
parser.add_argument('--item_keyword', action='store_true', help='item_keyword')
parser.add_argument('--image_fea', action='store_true', help='image_fea')
parser.add_argument('--image_fea_path', type=str, default='D:/Pytorch-extract-feature/MT/Res_embeddings/')
# parser.add_argument('--add_adjective', type=str2bool, default=False, help='item_keyword')
# parser.add_argument('--adjective_only', type=str2bool, default=True, help='item_keyword')
parser.add_argument('--Transformer', action='store_true', help='Transformer')
parser.add_argument('--padding', action='store_true', help='padding')

parser.add_argument('--teacher_mode', action='store_true', help='teaching_mode')
parser.add_argument('--use_adjective',action='store_true', help='False: no adjective; True: use the adjective')
parser.add_argument('--teacher_model_path', type=str, default='D:/PETER2/CSJ_output/attention_trans/AZ_CSJ_1/model1.pt', help='path for user_profile_path')

parser.add_argument('--item_keyword_path', type=str, default='./Amazon/Movies_and_TV/MT_keyword_one.pickle', help='path for loading item_keyword')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='decay_rate')
parser.add_argument('--ema_decay', type=float, default=0.95,
                    help='ema decay weight')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.7
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
student_model_path = os.path.join(args.checkpoint, 'student_model.pt')
prediction_path_teacher = os.path.join(args.checkpoint, args.outf_teacher)

# teacher_model_path = os.path.join(args.checkpoint, 'teacher_model.pt')

###############################################################################
# Load data
###############################################################################



class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


print(now_time() + 'Loading data')
corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size, args)
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
item2idx = corpus.item_dict.entity2idx
idx2item = corpus.item_dict.idx2entity
user_profile = corpus.user_profile
item_profile = corpus.item_profile
cls_idx = corpus.word_dict.word2idx['<cls>']
feature_set = corpus.feature_set
item_keyword_set = corpus.item_keyword_set
train_data = Batchify(corpus.train, word2idx, args, idx2item, args.words, args.key_len, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, word2idx, args, idx2item, args.words, args.key_len, args.batch_size)
test_data = Batchify(corpus.test, word2idx, args, idx2item, args.words, args.key_len, args.batch_size)

###############################################################################
# Build the model
###############################################################################
src_len_teacher = 2
src_len_student = 3
if args.use_feature:
    src_len_teacher = src_len_teacher + train_data.feature.size(1)
if args.item_keyword:
    src_len_teacher = src_len_teacher + train_data.item_k.size(1)
if args.use_adjective:
    src_len_teacher = src_len_teacher + train_data.adjective.size(1)


if args.image_fea:
    src_len_teacher += 1
    src_len_student += 1
# if args.user_item_profile:
#     src_len = src_len + train_data.user_profile.size(1)*train_data.user_profile.size(2) + train_data.item_profile.size(1)*train_data.item_profile.size(2)

tgt_len = args.words + 1  # added <bos> or <eos>
ntokens = len(corpus.word_dict)
nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
print('src_teacher: {},tgt :{}'.format(src_len_teacher,tgt_len))
print('src_student: {},tgt :{}'.format(src_len_student,tgt_len))
pad_idx = word2idx['<pad>']
d_k = args.emsize//args.nhead

# teacher_model = Transformer(args, args.peter_mask, src_len_teacher, tgt_len, src_vocab_size=ntokens, tgt_vocab_size=ntokens, d_model=args.emsize, n_layers=args.nlayers,n_heads=args.nhead,\
#                     d_k = d_k, d_v = d_k,d_ff =2048, dropout=args.dropout, pad_idx=pad_idx, nuser=nuser, nitem = nitem).to(device)

student_model = Transformer(args, args.peter_mask, src_len_student, tgt_len, src_vocab_size=ntokens, tgt_vocab_size=ntokens, d_model=args.emsize, n_layers=args.nlayers,n_heads=args.nhead,\
                    d_k = d_k, d_v = d_k,d_ff =2048, dropout=args.dropout, pad_idx=pad_idx, nuser=nuser, nitem = nitem).to(device)




ema = EMA(student_model, args.ema_decay)


text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()
img_criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

feature_mse = nn.MSELoss()
feature_mae = nn.L1Loss()


optimizer_student = torch.optim.SGD(student_model.parameters(), lr=args.lr)
# optimizer_teacher = torch.optim.SGD(teacher_model.parameters(), lr=args.lr)
if args.scheduler_lr:
    scheduler_student = torch.optim.lr_scheduler.StepLR(optimizer_student, 1, gamma=args.gamma)
    # scheduler_teacher = torch.optim.lr_scheduler.StepLR(optimizer_teacher, 1, gamma=0.25)

###############################################################################
# Training code
###############################################################################



class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):

        # f_s = f_s.view(bsz, -1)
        # f_t = f_t.view(bsz, -1)
        f_s= torch.transpose(f_s,0,1)
        f_t= torch.transpose(f_t,0,1)

        bsz = f_s.shape[0]
        G_s = torch.matmul(f_s, torch.transpose(f_s,1,2))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.matmul(f_t, torch.transpose(f_t,1,2))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss


Similarity_func = Similarity()


def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)

def adjust_learning_rate_cos(optimizer, epoch, step, len_epoch):
    # first 5 epochs for warmup
    global args
    warmup_iter = 5 * len_epoch
    current_iter = step + epoch * len_epoch
    max_iter = args.epochs * len_epoch
    lr = args.lr * (1 + math.cos(math.pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    if epoch < 5:
        lr = args.lr * current_iter / warmup_iter

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))







def train_one_epoch_student(data, epoch):

    student_model.train()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    clsf_loss = 0
    img_loss = 0.
    total_sample = 0
    ts_loss1 = 0
    ts_loss2 = 0
    TS_loss = 0
    while True:
        # batch-size * len
        user, item, rating, seq, seq_masked, feature, adjective, item_k, image_feature, user_profile, item_profile, cls = data.next_batch()  # (batch_size, seq_len), data.step += 1

        batch_size = user.size(0)
        user_profile = user_profile.view(batch_size,-1).to(device)
        item_profile = item_profile.view(batch_size,-1).to(device)
        cls_flag = torch.tensor([cls_idx] * (batch_size)).unsqueeze(1).to(device)
        if args.cosinelr:
            adjust_learning_rate_cos(optimizer_student, epoch, data.step, data.total_step)

        item_k = item_k.to(device)
        image_feature = image_feature.unsqueeze(1).to(device)

        feature = feature.to(device)  # (1, batch_size)

        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        seq = seq.to(device)  # (tgt_len + 1, batch_size)
        seq_masked = seq_masked.to(device)  # (tgt_len + 1, batch_size)
        cls = cls.to(device)



        index = cls == 1
        seq_pos = seq[index]
        rating_pos = rating[index]

        adjective = adjective.to(device)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer_student.zero_grad()

        enc_inputs = torch.cat([user_profile, item_profile],dim=1)


        dec_inputs_student = seq[:, :-1].to(device)
        dec_inputs_student = torch.cat([cls_flag, dec_inputs_student], dim=1)

        dec_inputs_student_masked = seq_masked[:, :-1].to(device)



        #
        log_word_prob_student_1, rating_p_student_1, log_context_dis_student_1, log_side_prob_1, hidden_student_1, logits_clsf,\
        log_word_prob_student_1_masked, rating_p_student_1_masked, log_context_dis_student_1_masked,log_side_prob_1_masked, hidden_student_1_masked \
            = student_model(user, item, enc_inputs, dec_inputs_student, dec_inputs_student_masked, image_feature, KD_T = False)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
        loss_clsf = criterion(logits_clsf, cls) # for sentence classification

        log_word_prob_student_1_pos = log_word_prob_student_1[index]
        log_context_dis_student_1_pos = log_context_dis_student_1[index]
        rating_p_student_1_pos = rating_p_student_1[index]
        context_dis = log_context_dis_student_1_pos.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # 复制了15 遍(batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)

        c_loss = text_criterion(context_dis.view(-1, ntokens), seq_pos[:, 1:-1].reshape((-1,)))
        r_loss = rating_criterion(rating_p_student_1_pos, rating_pos)
        t_loss = text_criterion(log_word_prob_student_1_pos.view(-1, ntokens), seq_pos[:,1:].reshape((-1,)))

        loss = args.text_reg * t_loss + args.context_reg * c_loss + args.rating_reg * r_loss + args.clsf_reg * loss_clsf



        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.clip)
        optimizer_student.step()

        #
        context_loss += batch_size * c_loss.item()
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        clsf_loss += batch_size * loss_clsf.item()
        # if args.image_fea:
        #     img_loss += batch_size * i_loss.item()
        # TS_loss += batch_size * loss_TS.item()

        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_c_loss = context_loss / total_sample
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            # cur_TS_loss = TS_loss / total_sample
            cur_cls_loss = clsf_loss / total_sample

            print(now_time() + 'Student: context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} |cls loss {:4.4f} | {:5d}/{:5d} batches'.format(
                    math.exp(cur_c_loss), math.exp(cur_t_loss), cur_r_loss, cur_cls_loss , data.step, data.total_step))

            context_loss = 0.
            text_loss = 0.
            rating_loss = 0.
            clsf_loss = 0
            ts_loss1 = 0
            ts_loss2 = 0
            TS_loss = 0
            img_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break


def evaluate_for_student(data):


    student_model.eval()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    img_loss = 0.
    total_sample = 0
    ts_loss1 = 0
    ts_loss2 = 0
    TS_loss = 0
    clsf_loss =0
    with torch.no_grad():
        while True:

            user, item, rating, seq, seq_masked, feature, adjective, item_k, image_feature, user_profile, item_profile, cls = data.next_batch()  # (batch_size, seq_len), data.step += 1

            batch_size = user.size(0)
            user_profile = user_profile.view(batch_size, -1).to(device)
            item_profile = item_profile.view(batch_size, -1).to(device)
            cls_flag = torch.tensor([cls_idx] * (batch_size)).unsqueeze(1).to(device)

            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.to(device)  # (tgt_len + 1, batch_size)
            seq_masked = seq_masked.to(device)  # (tgt_len + 1, batch_size)

            item_k = item_k.to(device)
            cls = cls.to(device)

            index = cls == 1
            seq_pos = seq[index]
            rating_pos = rating[index]

            image_feature = image_feature.unsqueeze(1).to(device)

            feature = feature.to(device)  # (1, batch_size)
            adjective = adjective.to(device)



            enc_inputs = torch.cat([user_profile, item_profile], dim=1)

            dec_inputs_student = seq[:, :-1].to(device)
            dec_inputs_student = torch.cat([cls_flag, dec_inputs_student], dim=1)

            dec_inputs_student_masked = seq_masked[:, :-1].to(device)

            log_word_prob_student_1, rating_p_student_1, log_context_dis_student_1, log_side_prob_1, hidden_student_1,logits_clsf, \
            log_word_prob_student_1_masked, rating_p_student_1_masked, log_context_dis_student_1_masked, log_side_prob_1_masked, hidden_student_1_masked \
                = student_model(user, item, enc_inputs, dec_inputs_student, dec_inputs_student_masked, image_feature,
                                KD_T=False)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            log_word_prob_student_1_pos = log_word_prob_student_1[index]
            log_context_dis_student_1_pos = log_context_dis_student_1[index]
            rating_p_student_1_pos = rating_p_student_1[index]

            context_dis = log_context_dis_student_1_pos.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # 复制了15 遍(batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)

            c_loss = text_criterion(context_dis.view(-1, ntokens), seq_pos[:, 1:-1].reshape((-1,)))
            r_loss = rating_criterion(rating_p_student_1_pos, rating_pos)


            t_loss = text_criterion(log_word_prob_student_1_pos.view(-1, ntokens), seq_pos[:, 1:].reshape((-1,)))

            loss_clsf = criterion(logits_clsf, cls)  # for sentence classification


            context_loss += batch_size * c_loss.item()
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            # TS_loss += batch_size * loss_TS.item()
            clsf_loss += batch_size * loss_clsf.item()

            # if args.image_fea:
            #     i_loss = img_criterion(image_fea_p.view(-1, 1), image_fea.view(-1, 1))
            #     img_loss += batch_size * i_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return context_loss / total_sample, text_loss / total_sample, rating_loss / total_sample, clsf_loss/ total_sample
    # return context_loss / total_sample, text_loss / total_sample, rating_loss / total_sample, img_loss / total_sample


def generate_for_student(data):
    # Turn on evaluation mode which disables dropout.
    student_model.eval()
    idss_predict = []
    context_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:

            user, item, rating, seq, seq_masked, feature, adjective, item_k, image_feature, user_profile, item_profile, cls= data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            seq = seq.to(device)  # (tgt_len + 1, batch_size)
            seq_masked = seq_masked.to(device)  # (tgt_len + 1, batch_size)
            cls = cls.to(device)
            user_profile = user_profile.view(batch_size, -1).to(device)
            item_profile = item_profile.view(batch_size, -1).to(device)
            cls_flag = torch.tensor([cls_idx] * (batch_size)).unsqueeze(1).to(device)

            item_k = item_k.to(device)
            image_feature = image_feature.unsqueeze(1).to(device)

            feature = feature.to(device)  # (1, batch_size)
            adjective = adjective.to(device)
            index = cls == 1
            seq_pos = seq[index]
            rating_pos = rating[index]
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)

            bos = seq[:,0].unsqueeze(1).to(device)  # (1, batch_size)


            token_flag = True



            enc_inputs = torch.cat([user_profile, item_profile], dim=1)

            dec_inputs_student = torch.cat([cls_flag, bos], dim=1)

            dec_inputs_student_masked = seq_masked[:, :-1].to(device)


            start_idx = dec_inputs_student.size(1)
            for idx in range(args.words):
                # produce a word at each step
                if idx == 0:

                    log_word_prob, rating_p, log_context_dis ,_,_, _, _, _, _, _, _= student_model(user, item, enc_inputs, dec_inputs_student, dec_inputs_student_masked, image_feature, token_flag,KD_T =False)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    rating_predict.extend(rating_p.tolist())
                    context = predict(log_context_dis, topk=args.words)  # (batch_size, words)
                    context_predict.extend(context.tolist())

                else:
                    # log_word_prob, log_context_dis, _, _ = model(user, item, text, False, False, False)
                    log_word_prob, rating_p, log_context_dis,_, _, _, _, _, _, _, _= student_model(user, item, enc_inputs, dec_inputs_student, dec_inputs_student_masked, image_feature, token_flag,KD_T =False)


                    # (batch_size, ntoken)


                word_prob = log_word_prob.exp().squeeze(1) # (batch_size, ntoken)
                # print(word_prob.size())
                word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
                # print(dec_inputs.size(),word_idx.size())
                dec_inputs_student = torch.cat([dec_inputs_student, word_idx.unsqueeze(1)],dim=1)  # (batch_size, seq_len)

                # dec_inputs = word_idx.unsqueeze(1)  # (len++, batch_size)
                # ids = torch.cat(dec_inputs) # (batch_size, seq_len)

            ids = dec_inputs_student[:,start_idx:].tolist()

            idss_predict.extend(ids)

            if data.step == data.total_step:
                break

    # rating
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'MAE {:7.4f}'.format(MAE))
    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    feature_batch = feature_detect(tokens_predict, feature_set)
    DIV = feature_diversity(feature_batch)  # time-consuming
    print(now_time() + 'DIV {:7.4f}'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, feature_set)
    print(now_time() + 'FCR {:7.4f}'.format(FCR))
    feature_test = [idx2word[i] for i in data.feature.squeeze(1).tolist()]  # ids to words
    FMR = feature_matching_ratio(feature_batch, feature_test)
    print(now_time() + 'FMR {:7.4f}'.format(FMR))
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    for (real, ctx, fake, r, r_p) in zip(text_test, tokens_context, text_predict, data.rating.tolist(), rating_predict):
        text_out += '{}\n{}\n{}\nrating_gt: {}               rating_p: {}\n\n'.format(real, ctx, fake, r, r_p)
    return text_out









def train_student():
    best_val_loss = float('inf')
    endure_count = 0
    for epoch in range(1, args.epochs_student + 1):
        print(now_time() + 'epoch {}'.format(epoch))
        train_one_epoch_student(train_data, epoch - 1)

        if epoch == 1:
            ema.register()

        ema.update()

        ema.apply_shadow()

        val_c_loss, val_t_loss, val_r_loss, clsf_loss= evaluate_for_student(val_data)

        if epoch % args.test_step==0:
            print(now_time() + 'Generating text')
            text_o = generate_for_student(test_data)
            name = str(epoch)+'_'+args.outf_student
            prediction_path_student = os.path.join(args.checkpoint, name)
            with open(prediction_path_student, 'w', encoding='utf-8') as f:
                f.write(text_o)
            print(now_time() + 'Epoch{}: Generated text saved to ({})'.format(epoch,prediction_path_student))


        if args.rating_reg == 0:
            val_loss = val_t_loss
        else:
            val_loss = val_t_loss + val_r_loss

        print(
            now_time() + 'Student: context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
                math.exp(val_c_loss), math.exp(val_t_loss), val_r_loss, val_loss))

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(student_model_path, 'wb') as f:
                torch.save(student_model.state_dict(), f)
        else:
            endure_count += 1
            print(now_time() + 'Endured {} time(s)'.format(endure_count))
            if endure_count == args.endure_times:
                print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                break
            if args.scheduler_lr:
                scheduler_student.step()
        print("第%d个epoch的recommender学习率：%f" % (epoch, optimizer_student.param_groups[0]['lr']))
        ema.restore()

        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        # scheduler.step()




train_student()

state_dict = torch.load(student_model_path)
student_model.load_state_dict(state_dict)
# Run on test data.


test_c_loss, test_t_loss, test_r_loss, clsf_loss = evaluate_for_student(test_data)
print('=' * 89)
print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | test | End of training'.format(
    math.exp(test_c_loss), math.exp(test_t_loss), test_r_loss))
prediction_path_student = os.path.join(args.checkpoint, args.outf_student)

print(now_time() + 'Generating text')
text_o = generate_for_student(test_data)
with open(prediction_path_student, 'w', encoding='utf-8') as f:
    f.write(text_o)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path_student))