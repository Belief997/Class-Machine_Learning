import os

##C:/Users/13480/Desktop/MyGit/Class-Machine_Learning/yolo/data/tiny_vid/bird/
path = input('请输入路径：')

f = os.listdir(path)
f1 = open('C:/Users/13480/Desktop/MyGit/Class-Machine_Learning/yolo/data/tiny_vid/train_dog.txt', 'a')
f2 = open('C:/Users/13480/Desktop/MyGit/Class-Machine_Learning/yolo/data/tiny_vid/bird_gt.txt')
n = 0

lines = f2.readlines()

for i in f:
    
    oldname = path + f[n]
    newname =  'train_dog_' + f[n]
    newpath = 'data/custom/images/' + newname + '\n'
    f1.write(newpath)
    os.rename(oldname, newname)
    print(oldname, '====>', newname)
    n += 1

f1.close()