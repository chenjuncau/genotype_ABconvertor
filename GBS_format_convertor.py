import pandas as pd
import os

# Change directory
os.chdir("local folder")

# Create a dictionary to map genotypes
genoCode = {}
with open("agriPlex_coding_map_illumnia.txt") as f:
    for line in f:
        Name, codeA, codeB, codeA2, codeB2 = line.strip().split('\t')
        genoCode[Name] = {codeA: codeA2, codeB: codeB2}

# Read the first line of the file to get genoList
with open('agriFlex_SNPlist.txt', 'r') as f:
    firstLine = f.readline().strip()
genoListTemp = firstLine.split("\t")
genoList = {idx: ele for idx, ele in enumerate(genoListTemp)}

# Process the genotype file
with open('ABFile.tsv', 'w') as oFile, open('agriFlex_SNPlist.txt', 'r') as inFile:
    # Write the first line to the output file
    oFile.write(inFile.readline())
    
    for line in inFile:
        vLst = line.strip().split("\t")
        geno = vLst[1:]
        tempList1 = [vLst[0]]
        
        for n, k in enumerate(geno):
            if k == "FAIL":
                tempList1.append("--")
                continue
            if len(k) == 1:
                tempList1.append(genoCode[genoList[n]][k] + genoCode[genoList[n]][k])
                continue
            if len(k) != 1:
                kk2 = k.strip().split(" / ")
                kk3temp = genoCode[genoList[n]][kk2[0]] + genoCode[genoList[n]][kk2[1]]
                if kk3temp == "BA":
                    kk3temp = "AB"
                tempList1.append(kk3temp)
        oFile.write('\t'.join(tempList1) + '\n')

# Read the processed file into a DataFrame and transpose it
df1_transposed = Database2.T

# Save the transposed DataFrame to a new file
df1_transposed.to_csv('python_Agriflex_matrixAB.txt', sep='\t', header=False, index=True)

# the end. 


import numpy as np

# Step 1: Read the text file into a numpy array
filename = 'input.txt'
data = np.loadtxt(filename)

# Step 2: Loop through the array and add specified values to each row
for i in range(data.shape[0]):
    if i == 0:
        data[i] += 1
    elif i == 1:
        data[i] += 2
    else:
        data[i] += 3

# Step 3: Write the modified array to a new file
output_filename = 'output.txt'
np.savetxt(output_filename, data, fmt='%f')



import numpy as np

# Step 1: Read the text file into a numpy array
filename = 'input.txt'
data = np.loadtxt(filename)

# Step 2: Loop through the array and add specified values to each row and column
for i in range(data.shape[0]):  # Iterate through rows
    for j in range(data.shape[1]):  # Iterate through columns
        if i == 0:
            data[i, j] += 1
        elif i == 1:
            data[i, j] += 2
        else:
            data[i, j] += 3
        
        if j == 0:
            data[i, j] += 3
        elif j == 1:
            data[i, j] += 4
        else:
            data[i, j] += 5

# Step 3: Write the modified array to a new file
output_filename = 'output.txt'
np.savetxt(output_filename, data, fmt='%f')



import pandas as pd

# Step 1: Read the text file into a pandas DataFrame
filename = 'input.txt'
data = pd.read_csv(filename, header=None, delim_whitespace=True)

# Step 2: Loop through the DataFrame and add specified values to each row and column
for i in range(data.shape[0]):  # Iterate through rows
    for j in range(data.shape[1]):  # Iterate through columns
        if i == 0:
            data.iloc[i, j] += 1
        elif i == 1:
            data.iloc[i, j] += 2
        else:
            data.iloc[i, j] += 3
        
        if j == 0:
            data.iloc[i, j] += 3
        elif j == 1:
            data.iloc[i, j] += 4
        else:
            data.iloc[i, j] += 5

# Step 3: Write the modified DataFrame to a new file
output_filename = 'output.txt'
data.to_csv(output_filename, header=False, index=False, sep=' ')



import json

# Step 1: Read the JSON file into a Python list of lists
filename = 'input.json'
with open(filename, 'r') as file:
    data = json.load(file)

# Step 2: Loop through the list of lists and add specified values to each row and column
for i in range(len(data)):  # Iterate through rows
    for j in range(len(data[i])):  # Iterate through columns
        if i == 0:
            data[i][j] += 1
        elif i == 1:
            data[i][j] += 2
        else:
            data[i][j] += 3
        
        if j == 0:
            data[i][j] += 3
        elif j == 1:
            data[i][j] += 4
        else:
            data[i][j] += 5

# Step 3: Write the modified list of lists back to a JSON file
output_filename = 'output.json'
with open(output_filename, 'w') as file:
    json.dump(data, file, indent=4)
    
    

lst = [11,18,9,12,23,4,17]
lost = []
for idx in range(len(lst)):
 val = lst[idx]
 if val > 15:
 lost.append(val)
 lst[idx] = 15
print("modif:",lst,"-lost:",lost)


#Looping through all key-value pairs
fav_numbers = {'eric': 17, 'ever': 4}
for name, number in fav_numbers.items():
 print(name + ' loves ' + str(number))
#Looping through all keys
fav_numbers = {'eric': 17, 'ever': 4}
for name in fav_numbers.keys():
 print(name + ' loves a number')
#Looping through all the values
fav_numbers = {'eric': 17, 'ever': 4}
for number in fav_numbers.values():
 print(str(number) + ' is a favorite')
 
    seen = set()
   output_list = []
   for item in input_list:
       if item not in seen:
           seen.add(item)
           output_list.append(item)
   return output_list


def var_method_0(x):
    n = len(x) # Number of samples
    ### BEGIN SOLUTION
    x_bar = sum(x) / n
    return sum([(x_i - x_bar)**2 for x_i in x]) / (n-1)



# 2877. Create a DataFrame from List

import pandas as pd

def createDataframe(student_data: List[List[int]]) -> pd.DataFrame:
   student_data=pd.DataFrame(student_data,columns =['student_id', 'age'])
   return student_data

# 2878. Get the Size of a DataFrame
import pandas as pd

def getDataframeSize(players: pd.DataFrame) -> List[int]:
    return [players.shape[0], players.shape[1]]

# 2879. Display the First Three Rows

import pandas as pd

def selectFirstRows(employees: pd.DataFrame) -> pd.DataFrame:
    return employees.head(3)
    
    

# 2880. Select Data

import pandas as pd

def selectData(students: pd.DataFrame) -> pd.DataFrame:
    students2=students.loc[students['student_id']==101]
    return students2[['name','age']]


# 2881. Create a New Column

import pandas as pd

def createBonusColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees['bonus']=employees['salary']*2
    return employees


# pandas cheatsheet. 

# keep the first duplicate value
import pandas as pd

def dropDuplicateEmails(customers: pd.DataFrame) -> pd.DataFrame:
    result=customers.drop_duplicates(subset=['email'], keep="first", inplace=False)
    return result

# remove the none value
import pandas as pd

def dropMissingData(students: pd.DataFrame) -> pd.DataFrame:
    #students=pd.DataFrame(students)
    result=students.dropna(subset=['name'])
    return result

# double one column value.
import pandas as pd

def modifySalaryColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees['salary']=employees['salary']*2
    return employees


# rename the column name.
import pandas as pd

def renameColumns(students: pd.DataFrame) -> pd.DataFrame:
    students=students.rename(columns={"id": "student_id", "first": "first_name", "last": "last_name", "age": "age_in_years"})
    return students

# change it to int for one columns value. 
import pandas as pd

def changeDatatype(students: pd.DataFrame) -> pd.DataFrame:
    students['grade'] = students['grade'].astype(int)
    return students


# fill na value. 
import pandas as pd

def fillMissingValues(products: pd.DataFrame) -> pd.DataFrame:
    products=products.fillna(0)
    return products

# concat two data frame table. 
import pandas as pd

def concatenateTables(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df3=pd.concat([df1, df2], ignore_index=True, axis=0)
    return df3

# pivot 
import pandas as pd

def pivotTable(weather: pd.DataFrame) -> pd.DataFrame:
    pivoted = weather.pivot(index="month", columns="city", values="temperature")
    return pivoted


import pandas as pd

def meltTable(report: pd.DataFrame) -> pd.DataFrame:
    result=pd.melt(report,id_vars=['product'],var_name='quarter',value_name='sales')
    return result


# subset one column and then sort by another column, and output one column. 
import pandas as pd

def findHeavyAnimals(animals: pd.DataFrame) -> pd.DataFrame:
    animal1 = animals[animals["weight"] > 100]
    animal2=animal1.sort_values(by=['weight'],ascending=False)
    return animal2[["name"]]







