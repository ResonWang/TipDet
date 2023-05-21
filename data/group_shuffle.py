import random

txt = "../dataset/SequenceInput_label_voc.txt"
f = open(txt)
lines = f.readlines()
f.close()
group = []
group_num = 4
total_shuffled = []
for line in lines:
    group.append(line)
    if len(group) == group_num:
        total_shuffled.append(group)
        group = []
random.shuffle(total_shuffled)

txt = "../dataset/SequenceInput_label_groupShuffle.txt"
f = open(txt,"w",newline='')
for item_group in total_shuffled:
    for item in item_group:
        f.write(item)
f.close()