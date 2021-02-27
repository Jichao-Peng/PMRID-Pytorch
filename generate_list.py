import os

input_path = " "
gt_path = " "
list_path = " "

f = open(list_path, 'w')
for root, tmp, files in os.walk(input_path, topdown=False):
    for name in files:
        if os.path.exists(gt_path+name):
            input_file = input_path+name
            gt_file = gt_path+name
            print(input_file + " " + gt_file)
            f.writelines(input_file + " " + gt_file + " - - -" + "\n")