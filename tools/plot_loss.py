import matplotlib.pyplot as plt
import numpy as np

log_file = sys.argv[1]
lines = open(log_file, 'r').readlines()

train_loss = []
test = [[] for _ in range(2)]
iter,iter_test = [],[]
i = 0
idx = [1,2]
while i < len(lines):
    line = lines[i]
    if 'solver.cpp:218] Iteration' in line:
        line = line.split()
        iter.append(int(line[5]))  # get iter num
        train_loss.append(float(line[-1]))  # get train loss
        i += 1
    elif 'Testing net (#0)' in line:
        line = line.split()
        #print(line)
        iter_test.append(int(line[5][:-1]))
        #while 'Restarting data prefetching from start.' in lines[i]:
        #    i += 1
        for n,j in enumerate(idx):
            #print 'line', i+j, '---', lines[i+j].split()[-1]
            test[n].append(float(lines[i+j].split()[-1]))
        #i += 9
        i += 1
    else:
        i += 1
plt.figure('train')
plt.plot(iter,train_loss)
plt.ylim([0,1])
plt.figure('test')
labels = ['acc/top-1','acc/top-5']
print 'iter_test',iter_test
print 'test',test
for i, t in enumerate(test):
    #print 'iter_test', iter_test
    #print 't', t
    plt.plot(iter_test,t,label=labels[i])
plt.legend()

for i, t in enumerate(test):
    t = np.asarray(t,dtype=np.float32)
    print(labels[i], iter_test[t.argmax()], t.max())

plt.show()
# print(iter)
# print(train_loss)
# print(iter_test)
# print(test)
