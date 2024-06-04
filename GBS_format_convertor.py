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
