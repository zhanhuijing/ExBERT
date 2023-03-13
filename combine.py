import pickle
import os
data=pickle.load(open('D:/PETER2/Amazon/Movies_and_TV/reviews.pickle','rb'))
index_dir = 'D:/PETER2/Amazon/Movies_and_TV/1'
org = []
Fea = []
item_ids = []
with open(os.path.join(index_dir, 'test.index'), 'r') as f:
    test_index = [int(x) for x in f.readline().split(' ')]
# W = open('test.txt','w')
for idx in test_index:
    org.append(data[idx]['template'][2]+'\n')
    Fea.append(data[idx]['template'][0]+' '+data[idx]['template'][1]+'\n')
    item_ids.append(data[idx]['item'])

import keybert
keywords1 = pickle.load(open('D:/PETER/Amazon/Movies and TV/MT_keyword_one.pickle','rb'))
keywords2 = pickle.load(open('D:/PETER/Amazon/Movies and TV/MT_keyword_two.pickle','rb'))

keywords3 = pickle.load(open('D:/PETER/Amazon/Movies and TV/MT_keyword_one_three.pickle','rb'))

keywords4 = pickle.load(open('D:/PETER/Amazon/Movies and TV/MT_keyword_one_four.pickle','rb'))


with open('key_dict/one/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0
    # org = []
    one_dict = []
    for line in f:
        cnt += 1
        # if cnt%4 == 1:
        #     # org.append(line)
        if cnt%4 == 3:
            one_dict.append(line)


with open('key_dict/two/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0
    # org = []
    two_dict = []
    for line in f:
        cnt += 1
        # if cnt%4 == 1:
        #     # org.append(line)
        if cnt%4 == 3:
            two_dict.append(line)


with open('key_dict/one_three/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0
    # org = []
    one_three_dict = []
    for line in f:
        cnt += 1
        # if cnt%4 == 1:
        #     # org.append(line)
        if cnt%4 == 3:
            one_three_dict.append(line)

with open('key_dict/one_four/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0
    # org = []
    one_four_dict = []
    for line in f:
        cnt += 1
        # if cnt%4 == 1:
        #     # org.append(line)
        if cnt%4 == 3:
            one_four_dict.append(line)


with open('key_word_fea/one/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0
    # org = []
    one = []
    for line in f:
        cnt += 1
        # if cnt%4 == 1:
        #     # org.append(line)
        if cnt%4 == 3:
            one.append(line)


with open('./key_word_fea/two/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0

    two = []
    for line in f:
        cnt += 1
        if cnt%4 == 3:
            two.append(line)


with open('./key_word_fea/one_three/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0

    one_three = []
    for line in f:
        cnt += 1
        if cnt%4 == 3:
            one_three.append(line)

with open('./key_word_fea/one_four/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0

    one_four = []
    for line in f:
        cnt += 1
        if cnt%4 == 3:
            one_four.append(line)



with open('./key_word/keyword_one_only/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0
    # org = []
    one_only = []
    for line in f:
        cnt += 1
        # if cnt%4 == 1:
        #     # org.append(line)
        if cnt%4 == 3:
            one_only.append(line)



with open('./key_word/keyword_two_only/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0
    # org = []
    two_only = []
    for line in f:
        cnt += 1
        # if cnt%4 == 1:
        #     # org.append(line)
        if cnt%4 == 3:
            two_only.append(line)




with open('./key_word/keyword_one_three_only/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0
    # org = []
    one_three_only = []
    for line in f:
        cnt += 1
        # if cnt%4 == 1:
        #     # org.append(line)
        if cnt%4 == 3:
            one_three_only.append(line)

with open('./key_word/keyword_one_four_only/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0
    # org = []
    one_four_only = []
    for line in f:
        cnt += 1
        # if cnt%4 == 1:
        #     # org.append(line)
        if cnt%4 == 3:
            one_four_only.append(line)



with open('Peter_fea/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0

    peter_ = []
    for line in f:
        cnt += 1
        if cnt%4 == 3:
            peter_.append(line)


with open('Peter_wo_fea/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0

    peter = []
    for line in f:
        cnt += 1
        if cnt%4 == 3:
            peter.append(line)

with open('D:/PETER2/key_word/AZ_Movies_TV.1.Nete.txt','r') as f:
    cnt = 0

    NETE = []
    for line in f:
        cnt += 1
        if cnt%4 == 3:
            NETE.append(line)


with open('D:/PETER2/adj_only/AZ_Movies_TV_adj_1/generated_fea_adj.txt','r') as f:
    cnt = 0

    # Fea = []
    adj=[]
    for line in f:

        if cnt%5 == 3:
            adj.append(line)
        cnt += 1
with open('fea_adj/AZ_Movies_TV_1/generated.txt','r') as f:
    cnt = 0

    # Fea = []
    fea_adj=[]
    for line in f:
        cnt += 1
        if cnt%4 == 3:
            fea_adj.append(line)


with open('img/img_only/AZ_Movies_TV_1/generated.txt', 'r') as f:
    cnt = 0

    # Fea = []
    img = []
    for line in f:
        cnt += 1
        if cnt % 4 == 3:
            img.append(line)



with open('img/img_fea/AZ_Movies_TV_1/generated.txt', 'r') as f:
    cnt = 0

    # Fea = []
    img_fea = []
    for line in f:
        cnt += 1
        if cnt % 4 == 3:
            img_fea.append(line)



with open('img/img_fea_key/one/AZ_Movies_TV_1/generated.txt', 'r') as f:
    cnt = 0

    # Fea = []
    img_fea_key_one = []
    for line in f:
        cnt += 1
        if cnt % 4 == 3:
            img_fea_key_one.append(line)



with open('img/img_fea_key/two/AZ_Movies_TV_1/generated.txt', 'r') as f:
    cnt = 0

    # Fea = []
    img_fea_key_two = []
    for line in f:
        cnt += 1
        if cnt % 4 == 3:
            img_fea_key_two.append(line)



with open('img/img_fea_key/one_three/AZ_Movies_TV_1/generated.txt', 'r') as f:
    cnt = 0

    # Fea = []
    img_fea_key_one_three = []
    for line in f:
        cnt += 1
        if cnt % 4 == 3:
            img_fea_key_one_three.append(line)





with open('img/img_fea_key/one_four/AZ_Movies_TV_1/generated.txt', 'r') as f:
    cnt = 0

    # Fea = []
    img_fea_key_one_four = []
    for line in f:
        cnt += 1

        if cnt % 4 == 3:
            img_fea_key_one_four.append(line)

with open('key_word_fea_one_four_length/20words/AZ_Movies_TV_1/generated.txt', 'r') as f:
    cnt = 0

    # Fea = []
    fea_key_one_four_20 = []
    for line in f:
        cnt += 1

        if cnt % 4 == 3:
            fea_key_one_four_20.append(line)


with open('key_word_fea_one_four_length/25words/AZ_Movies_TV_1/generated.txt', 'r') as f:
    cnt = 0

    # Fea = []
    fea_key_one_four_25 = []
    for line in f:
        cnt += 1

        if cnt % 4 == 3:
            fea_key_one_four_25.append(line)

with open('key_word_fea_one_four_length/30words/AZ_Movies_TV_1/generated.txt', 'r') as f:
    cnt = 0

    # Fea = []
    fea_key_one_four_30 = []
    for line in f:
        cnt += 1

        if cnt % 4 == 3:
            fea_key_one_four_30.append(line)

w = open('combine_generation.txt','w')
for i in range(len(org)):
    w.write('GT:                                '+org[i])
    w.write('Fea & Adj words:                   '+Fea[i])
    w.write("NETE:                              "+NETE[i])
    w.write("PETER:                             "+peter[i])
    w.write("PETER+:                            "+peter_[i])
    w.write('Different length of keywords with features(keywords do not be fed into transformer):\n')
    w.write("%-35s"%keywords1[item_ids[i]]+one_dict[i])
    w.write("%-35s"%keywords2[item_ids[i]]+two_dict[i])
    w.write("%-35s"%keywords3[item_ids[i]]+one_three_dict[i])
    w.write("%-35s"%keywords4[item_ids[i]]+one_four_dict[i])
    w.write('Different length of keywords with features(keywords fed into transformer):\n')
    w.write("%-35s"%keywords1[item_ids[i]]+one[i])
    w.write("%-35s"%keywords2[item_ids[i]]+two[i])
    w.write("%-35s"%keywords3[item_ids[i]]+one_three[i])
    w.write("%-35s"%keywords4[item_ids[i]]+one_four[i])
    w.write('Different length of keywords without features:\n')
    w.write("%-35s"%keywords1[item_ids[i]]+one_only[i])
    w.write("%-35s"%keywords2[item_ids[i]]+two_only[i])
    w.write("%-35s"%keywords3[item_ids[i]]+one_three_only[i])
    w.write("%-35s"%keywords4[item_ids[i]]+one_four_only[i])
    w.write("Only_Adj:                          "+adj[i])
    w.write("Fea_Adj:                           "+fea_adj[i])
    w.write('Image Modality:\n')
    w.write("Only_Img:                          "+img[i])
    w.write("Img_Fea:                           "+img_fea[i])
    w.write("Img_Fea_One:                       "+img_fea_key_one[i])
    w.write("Img_Fea_Two:                       "+img_fea_key_two[i])
    w.write("Img_Fea_One_three:                 "+img_fea_key_one_three[i])
    w.write("Img_Fea_One_four:                  "+img_fea_key_one_four[i])
    w.write('Different sen_len:(under one_four keywords with features)\n')
    w.write("15 words:                          " + one_four[i])
    w.write("20 words:                          " + fea_key_one_four_20[i])
    w.write("25 words:                          " + fea_key_one_four_25[i])
    w.write("30 words:                          " + fea_key_one_four_30[i])
    w.write('='*120+'\n')




    w.write('\n')
    w.flush()
w.close()

