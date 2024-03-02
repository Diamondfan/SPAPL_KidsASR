


text_dir = "/home/vijaysumaravi/Documents/database/OGI_Kids/docs/all.map"

words_list = {}
with open(text_dir, 'r') as rf, open('words.txt', 'w') as wf:
    line = rf.readline()
    while line:
        sen = line.strip().split(' ', 1)[1][1:-1]
        for word in sen.split(' '):
            word = word.strip(',')
            if word not in words_list:
                wf.write(word+'\n')
                words_list[word] = True
        line = rf.readline()


