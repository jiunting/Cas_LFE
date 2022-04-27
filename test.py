s = "xzxzx"

all_diff = []
num = 0
for i in range(len(s)-2):
    for j in range(i+1,len(s)-1):
        print('i,j=',i,j)
        a = s[:i+1]
        b = s[i+1:j+1]
        c = s[j+1:]
        print(' abc=',a,b,c)
        print('  ab=',a+b)
        print('  bc=',b+c)
        print('  ca=',c+a)
        if (a+b != b+c) or (a+b != c+a) or (b+c != c+a):
            print('add 1')
            all_diff.append(a+b+c)
            num += 1



np.matrix([[1,2,3,4,0],[5,6,7,8,1],[3,2,4,1,4], [4,3,5,1,6]])
