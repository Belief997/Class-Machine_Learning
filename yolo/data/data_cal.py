import os

##C:/Users/13480/Desktop/MyGit/Class-Machine_Learning/yolo/data/tiny_vid/bird/
path = input('请输入路径：')

f = os.listdir(path)
f2 = open('C:/Users/13480/Desktop/MyGit/Class-Machine_Learning/yolo/data/tiny_vid/bird_gt.txt')
n = 0

lines = f2.readlines()
write_dir = 'C:/Users/13480/Desktop/MyGit/Class-Machine_Learning/yolo/data/label/bird/'

for i in f:
    
    filename = f[n].replace('.JPEG', '.txt')
    newpath = write_dir + filename
    f1 = open(newpath, 'w')
    old_str = lines[n]
    old_str = old_str.split(' ')
    new_str = []
    new_str += '0'
    w = int(old_str[3]) / 128
    h = int(old_str[5]) / 128
    x = int(old_str[1]) / 128 + w / 2
    y = int(old_str[2]) / 128 + h / 2
    new_str += ' ' + str(x)
    new_str += ' ' + str(y)
    new_str += ' ' + str(w)
    new_str += ' ' + str(h)
    f1.write(new_str)
    n += 1
    f1.close()
