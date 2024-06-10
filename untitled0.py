# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:12:21 2024

@author: chenj
"""

    def romanToInt(self, s: str) -> int:
        m = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        
        ans = 0
        
        for i in range(len(s)):
            if i < len(s) - 1 and m[s[i]] < m[s[i+1]]:
                ans -= m[s[i]]
            else:
                ans += m[s[i]]
        
        return ans
        
    
grade_dicts={}
for i in range(len(grades)):
    if(i !=0):
      grade_dicts[grades[i][0]]={}
      grade_dicts[grades[i][0]][grades[0][1]]=int(grades[i][1])
      grade_dicts[grades[i][0]][grades[0][2]]=int(grades[i][2])
      grade_dicts[grades[i][0]][grades[0][3]]=int(grades[i][3])
      
      


def compress_vector(x):
    assert type(x) is list
    d = {'inds': [], 'vals': []}
    ###
    ### YOUR CODE HERE
    ###
    count=0
    count2=0
    for i in x:
        if (i != 0.0):
          d['inds'].append(count)
          d['vals'].append(i)
          count2=count2+1  
        count=count+1
    if(count2==0):
       d['inds']=[]
       d['vals']=[] 
    return d



if __name__ == '__main__':
    N = int(input())
    arr = []
    for _ in range(N):
        command,*line = input().split()
        num = list(map(int,line))
        if command == 'insert':
            arr.insert(num[0],num[1])
        elif command == 'print':
            print(arr)
        elif command == 'remove':
            arr.remove(num[0])
        elif command == 'append':
            arr.append(num[0])
        elif command == 'sort':
            arr.sort()
        elif command == 'pop':
            arr.pop()
        elif command == 'reverse':
            arr.reverse()
            
            
            
map(int, input().split())



def count_substring(string, sub_string):
    count = 0
    sub_len = len(sub_string)
    for i in range(len(string) - sub_len + 1):
        if string[i:i + sub_len] == sub_string:
            count += 1
    return count


        l = 0
        r = len(s) - 1
        while l < r:
            if not s[l].isalnum():
                l += 1
            elif not s[r].isalnum():
                r -= 1
            elif s[l].lower() == s[r].lower():
                l += 1
                r -= 1
            else:
                return False

        return True                    