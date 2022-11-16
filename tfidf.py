'''
Sua tarefa será gerar a matriz termo-documento usando TF-IDF por meio da aplicação das 
fórmulas  TF-IDF  na  matriz  termo-documento  criada  com  a  utilização  do  algoritmo  Bag of 
Words. Sobre o Corpus que recuperamos anteriormente. O entregável desta tarefa é uma 
matriz termo-documento onde a primeira linha são os termos e as linhas subsequentes são 
os vetores calculados com o TF-IDF. 
2. Sua tarefa será gerar uma matriz de distância, computando o cosseno do ângulo entre todos 
os vetores que encontramos usando o tf-idf. Para isso use a seguinte fórmula para o cálculo 
do  cosseno  use  a  fórmula  apresentada  em  Word2Vector  (frankalcantara.com) 
(https://frankalcantara.com/Aulas/Nlp/out/Aula4.html#/0/4/2) 
'''

import numpy as np
import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
from google.colab.data_table import DataTable
DataTable.max_columns = 4000

# Função responsável por filtrar e remover partes específicas da página.
def removerTags(soup):
    for data in soup(['style', 'script', 'head', 'header', 'meta', '[document]', 'title', 'footer', 'iframe', 'nav']):
        data.decompose()
        
    # Retornando todas as expressões em uma string, separando por espaço.
    return ' '.join(soup.stripped_strings)   


def filtrar(palavra):
    for letra in palavra:
      if letra.isalpha() == False:
        palavra = palavra.replace(letra, "")
    return palavra
  
  
# Carregando a biblioteca Spacy, baseando-se na língua inglesa.
nlp = spacy.load("en_core_web_sm")

# Variáveis para armazenar o link de cada página.
urls = ['https://hbr.org/2022/04/the-power-of-natural-language-processing',
        'https://www.sas.com/en_us/insights/analytics/what-is-natural-language-processing-nlp.html',
        'https://www.techtarget.com/searchenterpriseai/definition/natural-language-processing-NLP',
        'https://monkeylearn.com/natural-language-processing/',
        'https://en.wikipedia.org/wiki/Natural_language_processing']

# Definindo arrays para armazenar o resultado de todos os textos.
textoCompleto = []

# Loop para passar por cada url.
for url in urls:
  html = requests.get(url).text                     # Armazenando o conteúdo de toda a página.
  soup = BeautifulSoup(html, 'html.parser')         # Removendo parte da estrutura HTML do texto.
  texts = removerTags(soup)                         # Enviando o texto para filtrar.
  page = nlp(texts)

  # Armazenando as sentenças na lista geral.
  for sentence in page.sents:
    textoCompleto.append(sentence.text)

# Definição de uma variável para armazenar o vocabulário.
vocabulario = set()
for sentence in textoCompleto:
  for palavra in sentence.split():
      palavraFinal = filtrar(palavra.strip())
      if palavraFinal != "":
        vocabulario.add(palavraFinal)
vocabulario = sorted(vocabulario)

# Definindo duas listas para armazenar o resultado.
bagOfWords = []
tfidf = []

# Adicionando o vocabulário nas listas, para aparecer como índice na tabela.
bagOfWords.append(vocabulario)
tfidf.append(vocabulario)

contador = 1
dicionarioIndiceSentenca = {}

# Loop responsável por popular a lista do Bag Of Words
# Responsável, também, por popular um dicionário com a relação entre uma frase
# e a quantidade de termos nela.
for sentence in textoCompleto:
  tamanhoSentenca = len(sentence)
  vetor = [0] * len(vocabulario)
  for palavra in sentence.split():
    palavraFinal = filtrar(palavra.strip())
    if palavraFinal != "":
      vetor[vocabulario.index(palavraFinal)] += 1
  bagOfWords.append(vetor)

  dicionarioIndiceSentenca[contador] = tamanhoSentenca
  contador += 1


colunaPalavra = 1
# Dicionário que irá armazenar a quantidade de documentos que tem tal palavra.
dicionarioPosicaoPalavra = {}

for j in range(len(bagOfWords[0])):
  dicionarioPosicaoPalavra[colunaPalavra] = 0

  for i in range(1, len(bagOfWords)):
    if bagOfWords[i][j] != 0:
      dicionarioPosicaoPalavra[colunaPalavra] += 1
  colunaPalavra += 1

# Obtendo todas as informações dos dicionários e realizando o cálculo do TFIDF.
for i in range(1, len(bagOfWords)):
  tamanhoSentenca = dicionarioIndiceSentenca[i]
  vetor = []
  for x in range(len(bagOfWords[i])):
    digito = bagOfWords[i][x]
    if digito != 0:
      qtdDocumentos = len(bagOfWords)
      qtdDocumentosQueTemAPalavra = dicionarioPosicaoPalavra[x+1]
      digito = digito / tamanhoSentenca * np.log((qtdDocumentos / qtdDocumentosQueTemAPalavra))
    
    vetor.append(digito)
  tfidf.append(vetor)

np.seterr(all = "raise")
matrizCosseno = []

# Comparando cada vetor com os outros da matriz e realizando o cálculo do cosseno.
for x in range(1, len(tfidf)):
  vetor = []
  for i in range(1, len(tfidf)):
    try:
      cos_sim = np.dot(tfidf[x], tfidf[i])/(np.linalg.norm(tfidf[x])*np.linalg.norm(tfidf[i]))
    except:
      cos_sim = 0
    vetor.append(cos_sim)
  matrizCosseno.append(vetor)

dataframe1 = pd.DataFrame(tfidf)
dataframe1.drop([0], inplace=True)
dataframe1.set_axis(vocabulario, axis='columns', inplace=True)
dataframe2 = pd.DataFrame(matrizCosseno)
display(dataframe1)
display(dataframe2)
