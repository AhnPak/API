import pandas as pd
from tqdm import tqdm
import chardet
import os
import unicodedata

def run():
    dir_path = 'RISS'
    data = dict()
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            L = file_path.split('/')
            uni = unicodedata.normalize('NFC', L[len(L) - 1])
            if (unicodedata.normalize('NFC', '한국어') in uni):

                with open(file_path, 'rb') as rawdata:
                    result = chardet.detect(rawdata.read(10000))

                if result['encoding'] == 'ascii':
                    continue
                if result['encoding'] == 'EUC-KR':
                    encoding_option = 'cp949'
                else:
                    encoding_option = result['encoding']

                df = pd.read_csv(file_path, names=['name', 'abstract', 'author', 'year', 'link'], sep=',', encoding=encoding_option)
                df = df.dropna()
                category = uni.split('.')[0]
                category = category.replace('_한국어', '')
                for index, row in tqdm(df.iterrows(), total=len(df)):
                    name = row['name']
                    abstract = row['abstract']
                    author = row['author']
                    year = row['year']
                    link = row['link']
                    if(name not in data):
                        data[name] = (name, abstract, author, year, link, category)
    return data