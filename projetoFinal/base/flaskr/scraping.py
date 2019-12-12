from urllib.request import FancyURLopener
from bs4 import BeautifulSoup
import re
import pandas as pd
import os.path
import nltk
import collections
import math
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.exceptions import abort
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import io
import random
from flask import Flask, Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG


from matplotlib.figure import Figure


from flaskr.auth import login_required
from flaskr.db import get_db
from flask import Flask
#nltk.download('stopwords')
#nltk.download('punkt')


class MyOpener(FancyURLopener):
    version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
myopener = MyOpener()

BASE_DIV = "brand-complaint-list-item-anchor"
PREVIEW_TITLE = "brand-complaint-list-item-title"
PREVIEW_CATEGORY = "brand-complaint-list-item-date-category"
PREVIEW_TEXT = "brand-complaint-list-item-description"
PREVIEW_ID = "brand-complaint-list-item-details"
PREVIEW_STATE = "complaints-list-status-holder"
FULL_TEXT = "complaint-detail-body-description"


BASE_LINK ="https://portaldaqueixa.com/brands/"
COMPLAINS = "/complaints"
CSV = ".csv"

bp = Blueprint("scraping", __name__, url_prefix="/scraping")

class Complain:
  def __init__(self, brand, idd, url, title, category, previewText, state):
    self.brand = brand
    self.idd = idd
    self.url = url
    self.title = title
    self.category = category
    self.previewText = previewText
    self.fullText = None
    self.state = state

  def getExtraInfo(self):
    extraInfoObject = soupIt(self.url)
    self.fullText = extraInfoObject.find_all('div', {"class": FULL_TEXT})[0].getText()

  def getDict(self):
    return { 
             "id": self.idd,
             "brand" : self.brand,
             "url": self.url,
             "title" : self.title,
             "category": self.category,
             "previewText": self.previewText,
             "fullText": self.fullText,
             "state": self.state}

  def saveInDataBase(self):
    db = get_db()
    db.execute(
        "INSERT INTO complain (idd, brand, url, title, category, previewText, fullText, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (self.idd, self.brand, self.url, self.title, self.category, self.previewText, self.fullText, self.state))
    db.commit()
  



@bp.route("/scrapBrand", methods=("GET", "POST"))
def scrapBrand():
    """Log in a registered user by adding the user id to the session."""
    if request.method == "POST":
        brand = request.form["brand"]
        nPages = request.form["numberOfPages"]
        sPage = request.form["startingPage"]
        nPages = int(nPages)
        sPage = int(sPage)

        db = get_db()
        error = None
        complains = scrapBrandSavingFile(brand, tryToLoadFile=True, startingPage=sPage,pages=nPages)
        #complains = scrapComplainsFromBrand(brand, startingPage=1, pages=2, fullInfo=True, dictionary=False, debug=True)
        for complain in complains:
          complain.saveInDataBase()

        if error is None:
            return redirect(url_for("index"))

        flash(error)

    return render_template("scraping/scrapBrand.html")


@bp.route("/<string:id>/brand", methods=['GET', 'POST'])
def generateTdIdfWEB(id):
    """Show all the complains, most recent first."""

    df = simplifyTextInFile(id)
    df = generateTdIdf(id,df)
    db = get_db()
    complains = db.execute(
        "SELECT *"
        " FROM complain"
        " WHERE brand = ?" 
        " ORDER BY idd DESC",
            (id,),
    ).fetchall()

    output = io.BytesIO()


    names = []
    if request.method == 'POST':
      if request.form['submit_button'] == '1 Theme':
        figs = labelComents(id, df, 1)
      elif request.form['submit_button'] == '2 Theme':
        figs = labelComents(id, df, 2)
      elif request.form['submit_button'] == '3 Theme':
        figs = labelComents(id, df, 3)
      elif request.form['submit_button'] == '4 Theme':
        figs = labelComents(id, df, 4)
      elif request.form['submit_button'] == '5 Theme':
        figs = labelComents(id, df, 5)

      print(figs[0])
      #FigureCanvas(figs[0]).print_png(output)
      #figs[0].to_image().save(output, 'PNG')
      #output.seek(0)

      #output.savefig("./media/"+final_name)
      
      names = []
      for i in range(0,len(figs)):
        fig = figs[i]
        name = str(random.randint(0, 100000))+'.png'
        fig.to_file('./flaskr/static/'+name)
        names+=[name]


    return render_template("brand/brandpage.html", brandName=id, complains=complains, figs=names)




def soupIt(urll):
  html = myopener.open(urll).read()
  soup = BeautifulSoup(html,"lxml")
  return soup


def openBrand(brand, page=1):
  if page==1:
    urll = BASE_LINK + brand + COMPLAINS
  else:
    urll = BASE_LINK + brand + COMPLAINS +"?p=" + str(page)

  html = myopener.open(urll).read()
  soup = BeautifulSoup(html,"lxml")
  return soup


def scrapComplainsFromBrand(brand, startingPage=1, pages=10, fullInfo=False, dictionary=False, debug=True):
  allComplains = []
  finalPage = pages+startingPage
  for page in range(startingPage,finalPage):
    
    if debug:
        print("Page: " + str(page) + "/" + str(finalPage-1))

    pageComplains = scrapBrandPage(brand, page, fullInfo, dictionary)
    allComplains += pageComplains
  
  return allComplains


def scrapBrandPage(brand, pageNumber, fullInfo=False, dictionary=False, debug=True):
  soup = openBrand(brand, pageNumber)
  complains = soup.findAll("a", {"class": BASE_DIV})
  complainsArray = []

  for complain in complains:
    c = extractComplainFromRaw(brand, complain)
    
    if fullInfo:
        if debug:
            print("\t id: " + c.idd)

        c.getExtraInfo()
    if dictionary:
      complainsArray += [c.getDict()]
    else:
      complainsArray += [c]

  return complainsArray


def extractComplainFromRaw(brand, complainObject):
  urll = complainObject['href']
  title = complainObject.find_all('h3', {"class": PREVIEW_TITLE})[0].getText()
  category = complainObject.find_all('span', {"class": PREVIEW_CATEGORY})[0].getText()
  previewText = complainObject.find_all('div', {"class": PREVIEW_TEXT})[0].getText()
  idd = complainObject.find_all('div', {"class": PREVIEW_ID})[0].h5.getText()[1:]
  state = complainObject.find_all('div', {"class": PREVIEW_STATE})[0].span.getText()
  return Complain(brand, idd ,urll,title,category,previewText,state)


def wasBrandPreviouslyLoaded(brand):
    return os.path.isfile(brand + CSV) 


def loadDataFrame(brand):
    df = pd.read_csv(brand+CSV)
    df = df.set_index('id')
    return df

def saveDataFrame(brand, df):
    df.to_csv(brand+CSV)


#If there is a file on the same directory the results will be concatenated and last ones added will be deletede
def scrapBrandSavingFile(brand, tryToLoadFile=True, startingPage=1,pages=25, returnTypeDF=False):
    
    if tryToLoadFile:
        wasThereAFile = wasBrandPreviouslyLoaded(brand)

    
    if tryToLoadFile and wasThereAFile:
        originalDF = loadDataFrame(brand)


    newData = scrapComplainsFromBrand(brand, startingPage=startingPage, pages=pages, fullInfo=True, dictionary=returnTypeDF)
    

    dd = []
    for complain in newData:
      dd += [complain.getDict()]

    newDataDF = pd.DataFrame(dd, columns=['id','brand', 'category', 'url', 'state','title','previewText','fullText'])
    newDataDF = newDataDF.set_index('id')

    if tryToLoadFile and wasThereAFile:
        finalDF = pd.concat([originalDF, newDataDF])
        finalDF = finalDF.drop_duplicates()
    else:
        finalDF = newDataDF

    saveDataFrame(brand, finalDF)


    if returnTypeDF:
      return finalDF
    else:
      return newData





### TEXT CLEANING


def simplifyText(text, brand):
  words = re.findall(r'\w+', text.lower())
  text = ' '.join(words)
  text = text.replace(brand+' ', '')
  text = text.replace(' wi fi ', ' wifi ')
  return removeStopWords(text)


def simplityTextOfDF(brand, df):
  df['fullText'] = df['fullText'].apply(simplifyText, brand=brand)
  df['previewText'] = df['previewText'].apply(simplifyText, brand=brand)
  df['title'] = df['title'].apply(simplifyText, brand=brand)
  return df


def simplifyTextInFile(brand):
  df = loadDataFrame(brand)
  df = simplityTextOfDF(brand, df)
  saveDataFrame(brand, df)
  return df


def removeStopWords(text):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    words = text.split()
    cleanWords = []
    for word in words:
      if word not in stopwords:
        cleanWords += [word]

 
    return ' '.join(cleanWords)



##IDF and tdIDF


def idf(counter, quantityOfComents):
    coeficient = quantityOfComents / counter
    operation = math.log((coeficient),2)
    return operation

def tdIdf(coment,mostSaidWords):
    vec_tf = np.zeros(100, dtype=float)
    counter =  collections.Counter(coment.split())
    for w in counter.keys():
        if w in mostSaidWords.index:
            val = mostSaidWords.loc[w]['idf'] * counter[w] 
            try:
                idx = mostSaidWords.index.get_loc(w)
                vec_tf[idx] = val
            except:
                continue
            
    return vec_tf

def normalizeTdInf(df):
  X1 = df['tdIdf'].tolist()
  scaler = StandardScaler()  
  scaler.fit(X1)
  X1 = scaler.transform(X1)
  neww = []
  for el in X1:
    neww += [np.array(el)]
  df['normalizedTdInf'] = neww
  return df


def generateTdIdf(brand,df, limit=100):
    print(df.iloc[0])
    quantityOfComents = len(df)
    text = ' '.join(df['fullText'])
    words = text.split()

    counters = collections.Counter(words)
    dfWords = pd.DataFrame.from_dict(counters, orient = 'index', columns=['counter'])
    dfWords = dfWords.sort_values(by='counter', ascending=False)

    mostSaidWords = dfWords.iloc[0:limit]

    mostSaidWords['idf'] = mostSaidWords['counter'].apply(idf, quantityOfComents=quantityOfComents)

    df['tdIdf'] = df['fullText'].apply(tdIdf, mostSaidWords=mostSaidWords)
    df = normalizeTdInf(df)
    saveDataFrame(brand, df)

    return df



### Label commnets




#Contar elementos por cluster del modelo


def CrearKMeans(k,df):
    lista_vect = df['normalizedTdInf'].tolist()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(lista_vect)
    lb_km = kmeans.labels_
    return lb_km, kmeans


def e_x_cluster(lb_km):
    e_clus = collections.Counter(lb_km)
    cant_x_cl = pd.DataFrame.from_dict(e_clus, orient = 'index')
    cant_x_cl.rename(columns={0:'Cantidad_elementos'}, inplace=True)
    cant_x_cl = cant_x_cl.sort_index(axis=0)
    return cant_x_cl

def cant_palabras_clust(vec_etiqueta,i):
    vec_coments = vec_etiqueta[vec_etiqueta['cluster'] == i]
    coms_k_i = vec_coments['fullText'].tolist()
    allText = ' '.join(coms_k_i).split()
    pal_k_i = collections.Counter(allText)
    df_pal_k_i = pd.DataFrame.from_dict(pal_k_i, orient = 'index')
    df_pal_k_i.rename(columns={0:'freq_pal'}, inplace=True)
    df_pal_k_i = df_pal_k_i.sort_values('freq_pal',ascending=False)
    
    return df_pal_k_i 

  
def cont_x_clus(labels, x_clus):
    palabras_cl = []
    cnt_labels = len(set(labels))
    for i in range (cnt_labels):
        palabras_cl.append(cant_palabras_clust(x_clus,i))
    return palabras_cl
  

def word_cloud(pl_x_cl,i):
  dict_p=pl_x_cl[i].to_dict()
  wordcloud = WordCloud(background_color="white",max_words=10)
  wordcloud.generate_from_frequencies(frequencies=dict_p['freq_pal'])
  #plt.figure()
  #plt.title('Cluster ['+str(i)+"] : "+pl_x_cl[i].index[0]+" "+pl_x_cl[i].index[1]+" "+pl_x_cl[i].index[2]+" "+pl_x_cl[i].index[3]+" "+pl_x_cl[i].index[4]+" "+pl_x_cl[i].index[5])
  #plt.imshow(wordcloud, interpolation="bilinear")
  #plt.axis("off")
  #plt.show()

  return wordcloud

def labelComents(brand, df, nClusters):

    labels , modelo = CrearKMeans(nClusters,df)
    cant_x_cluster = e_x_cluster(labels)

    df['cluster'] = labels
    saveDataFrame(brand,df)
    df.head()
    palabrasx_cls = cont_x_clus(labels,df)

    nubes = []
    cnt_labels = len(set(labels))
    for i in range(0,cnt_labels):
        nubes.append(word_cloud(palabrasx_cls,i))

    return nubes




#MAIN

def main(brand):
    simplifyTextInFile(brand)
    df = generateTdIdf(brand)
    labelComents(brand, df,5)

     