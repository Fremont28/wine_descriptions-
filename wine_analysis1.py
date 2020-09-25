#5/14/19

#This script was created to find patterns and trends in wine data 
#mainly looking at the text (wine descriptions) of over 8,000 wines
#across the world ranging from the US to Australia. 

#import libraries 
import numpy as np 
import pandas as pd 
import re 
from collections import Counter 
import nltk 
from nltk.corpus import stopwords
import numbers
from difflib import SequenceMatcher
from nltk.corpus import stopwords 
from collections import Counter
import re 
import numpy as np 
import pandas as pd 
import math 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import fileinput
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

#import csv files 
df1=pd.read_csv("por_wines1.csv",encoding="latin-1") 
df1['pais']="Por"
df2=pd.read_csv("wines_desc_spain.csv",encoding="latin-1") 
df2['pais']="Esp"
df3=pd.read_csv("wines_desc_us1.csv",encoding="latin-1") 
df3['pais']="US"
df4=pd.read_csv("wines_desc_us2.csv",encoding="latin-1") 
df4['pais']="US"
df5=pd.read_csv("wines_desc_us3.csv",encoding="latin-1") 
df5['pais']="US"
df6=pd.read_csv("wines_desc_us4.csv",encoding="latin-1") 
df6['pais']="US"
df7=pd.read_csv("wines_desc_us5.csv",encoding="latin-1") 
df7['pais']="US"
df8=pd.read_csv("wines_desc_us6.csv",encoding="latin-1") 
df8['pais']="US"
df9=pd.read_csv("wines_desc_us7.csv",encoding="latin-1") 
df9['pais']="US"
df10=pd.read_csv("wines_desc_us8.csv",encoding="latin-1") 
df10['pais']="US"
df11=pd.read_csv("wines_desc_us9.csv",encoding="latin-1") 
df11['pais']="US"
df12=pd.read_csv("wines_desc_us10.csv",encoding="latin-1") 
df12['pais']="US"
df13=pd.read_csv("wines_desc_us11.csv",encoding="latin-1")  
df13['pais']="US"
df14=pd.read_csv("wines_desc_fr1.csv",encoding="latin-1") 
df14['pais']="Fra"
df15=pd.read_csv("wines_desc_fr2.csv",encoding="latin-1") 
df15['pais']="Fra"
df16=pd.read_csv("wines_desc_fr3.csv",encoding="latin-1") 
df16['pais']="Fra"
df17=pd.read_csv("wines_desc_fr4.csv",encoding="latin-1") 
df17['pais']="Fra"
df18=pd.read_csv("wines_desc_fr5.csv",encoding="latin-1") 
df18['pais']="Fra"
df19=pd.read_csv("wines_desc_fr6.csv",encoding="latin-1") 
df19['pais']="Fra"
df20=pd.read_csv("wines_desc_fr7.csv",encoding="latin-1") 
df20['pais']="Fra"
df21=pd.read_csv("wines_desc_fr8.csv",encoding="latin-1") 
df21['pais']="Fra"
df22=pd.read_csv("wines_desc_it1.csv",encoding="latin-1") 
df22['pais']="Ita"
df23=pd.read_csv("wines_desc_arg1.csv",encoding="latin-1") 
df23['pais']="Arg"
df24=pd.read_csv("wines_desc_aus1.csv",encoding="latin-1") 
df24['pais']="Aus"
df25=pd.read_csv("wines_desc_car1.csv",encoding="latin-1") 
df25['pais']="Car"
df26=pd.read_csv("wines_desc_chile.csv",encoding="latin-1") 
df26['pais']="Chile"
df27=pd.read_csv("wines_desc_ger1.csv",encoding="latin-1") 
df27['pais']="Ger"
df28=pd.read_csv("wines_desc_gre1.csv",encoding="latin-1") 
df28['pais']="Gre"
df29=pd.read_csv("wines_desc_ire1.csv",encoding="latin-1") 
df29['pais']="Ire"
df30=pd.read_csv("wines_desc_eng1.csv",encoding="latin-1") 
df30['pais']="Eng"
df31=pd.read_csv("wines_desc_jp1.csv",encoding="latin-1") 
df31['pais']="JP"
df32=pd.read_csv("wines_desc_mex1.csv",encoding="latin-1") 
df32['pais']="Mex"
df33=pd.read_csv("wines_desc_nz1.csv",encoding="latin-1") 
df33['pais']="NZ"
df34=pd.read_csv("wines_desc_sco1.csv",encoding="latin-1") 
df34['pais']="Sco"
df35=pd.read_csv("wines_desc_sa1.csv",encoding="latin-1") 
df35['pais']="SA"

#combine country dataframes 
fin_vinos=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,
df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,
df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,
df32,df33,df34,df35],axis=0)

fv1=pd.read_csv("fv1.csv",encoding="latin-1", lineterminator='\n')
fv1['desc']=fv1['desc'].str.lower()
fv1=fv1[(fv1.year>=1000) & (fv1.price>5)] 

#fill is missing values with median of column 
fv1=fv1.fillna(fv1.median())
fv1=fv1[~fv1.pais.str.contains("JP")] #drop japan wines
fv1=fv1[~fv1.pais.str.contains("Eng")] #drop england wines 
fv1=fv1[~fv1.pais.str.contains("Car")]
fv1=fv1[~fv1.pais.str.contains("Ire")]

#average price and year by country 
avg_pais=fv1.groupby('pais')[["year","price"]].median()
avg_pais.sort_values(by="price") 

#outlier prices 
group_thresh=fv1.groupby('pais')['price'].apply(lambda c: (c>50).sum()/len(c))
group_thresh.sort_values() #61.2% france wines above $50 usd, 54.3% por, us 48.9% (gre 0.0%, nz 4.8%, aus 9.6%)

#Prices by region and year range 
bins=[1000,1950,1970,1990,2010,2015,2018] #create year bibs 
fv1['bin_year']=pd.cut(fv1['year'],bins)
avg_yr_prices=fv1.groupby('bin_year')['price'].mean() 

#percent of wines by country in binned year 
year_pais=fv1.groupby(['pais','bin_year']).count()
year_pais=year_pais.dropna()
year_pais=year_pais['year']
year_pais=pd.DataFrame(year_pais)

#count year bins by country 
cnt_pais=fv1.groupby('pais').count()['bin_year']
cnt_pais=pd.DataFrame(cnt_pais)

#percent (year bins) by country 
wine_yr_pcts = year_pais.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
wine_yr_pcts.sort_values(by="year")

#correlation between year and price 
corX=fv1[fv1.year>=1900]
corX['year'].corr(corX['price'])  

#most expensive wines  
exp_vinos=fv1[fv1.price>=99]
exp_vinos=exp_vinos[["pais","price","year"]]
exp_vinos.set_index('pais',inplace=True)

#percent by bin year groups 
bin_yrs=fv1.groupby('bin_year')['year'].count() 
bin_yrs=bin_yrs.reset_index()
bin_yrs['sum']=bin_yrs['year'].sum()
bin_yrs['pct']=(bin_yrs['year'])/(bin_yrs['sum'])
bin_yrs


#****************************************
#TEXT ANALYSIS 
#I. 
listos_wines=fv1 #~8,600 wine description rows 
listos_wines=listos_wines[listos_wines.pais!="US"] #non-US wines 
listos_wines=listos_wines['desc'].tolist()
listos_wines=' '.join(listos_wines)
listos_wines=listos_wines.split(' ')
listos_wines=[x.lower() for x in listos_wines]
listos_wines = list(filter(None, listos_wines))
lw=' '.join(listos_wines)
lw1=lw
lw1=lw1.lower()
lw1=re.sub(r'\W+', ' ', lw1)
list_ana=lw1.split(' ')
list_ana=list(filter(None,list_ana))


#clean and then count word frequency 
list_ana = [word for word in list_ana if word not in stopwords.words('english')] #remove stopwords 
list_ana=[i for i in list_ana if len(i) > 1]
str_ana=' '.join(list_ana)
str_ana=str_ana.split()
str_ana=[i for i in str_ana if len(i) >= 2]
fruits_cnt=Counter(str_ana).most_common(250) #250 most frequent palabras? 
fruits=pd.DataFrame(fruits_cnt) 
fruits.columns=['palabra','count']
fruits=fruits.iloc[1:fruits.shape[0]]
fruits=fruits[fruits['count']>=5]
word1="parker" #eliminate common words in the wine descriptions 
word2="san"
word3="francisco"
word4="points" 
word5="redwood"
word6="city"
word7="read"
word8="2017"
word9="2018"
word10="2019"
word11="2011"
word12="2015"
word13="2016"
word14="hollywood"
word15="wines"
word16="wine"
word17="robert"
word18="advocate"
word19="james"
fruits = fruits[~fruits["palabra"].str.contains(word1)] 
fruits = fruits[~fruits["palabra"].str.contains(word2)]
fruits = fruits[~fruits["palabra"].str.contains(word3)]
fruits = fruits[~fruits["palabra"].str.contains(word4)]
fruits = fruits[~fruits["palabra"].str.contains(word5)] 
fruits = fruits[~fruits["palabra"].str.contains(word6)]
fruits = fruits[~fruits["palabra"].str.contains(word7)]
fruits = fruits[~fruits["palabra"].str.contains(word8)]
fruits = fruits[~fruits["palabra"].str.contains(word9)] 
fruits = fruits[~fruits["palabra"].str.contains(word10)]
fruits = fruits[~fruits["palabra"].str.contains(word11)]
fruits = fruits[~fruits["palabra"].str.contains(word12)]
fruits = fruits[~fruits["palabra"].str.contains(word13)]
fruits = fruits[~fruits["palabra"].str.contains(word14)]
fruits = fruits[~fruits["palabra"].str.contains(word15)]
fruits = fruits[~fruits["palabra"].str.contains(word16)]
fruits = fruits[~fruits["palabra"].str.contains(word17)]
fruits = fruits[~fruits["palabra"].str.contains(word18)]
fruits = fruits[~fruits["palabra"].str.contains(word19)]
fruits['pct_palabra']=fruits['count']/(sum(fruits['count']))
fruits 
fruits.to_csv("mundo_todos.csv") #convert to csv file 


#II. Words before and after a specific wine word in the wine 
#cleaning text 
pais1=fv1
pais1=pais1[pais1.pais=="Fra"] #country 
listos_winesX=pais1['desc'].tolist() 
listos_winesX=' '.join(listos_winesX)
listos_winesX=listos_winesX.split(' ')
listos_winesX=[x.lower() for x in listos_winesX]
listos_winesX = list(filter(None, listos_winesX))
lwXX=' '.join(listos_winesX)
lwX1=lwXX
lwX1=lwX1.lower()
lwX1=re.sub(r'\W+', ' ', lwX1) 
list_anaX=lwX1.split(' ')
list_anaX=list(filter(None,list_anaX))
list_anaX = [word for word in list_anaX if word not in stopwords.words('english')]
list_anaX=[i for i in list_anaX if len(i) > 1]
list_anaX=[x.lstrip('0123456789) ') for x in list_anaX]
list_anaX = list(filter(None, list_anaX)) 
str_ana=' '.join(list_anaX)
str_ana=str_ana.split()
str_ana1=[i for i in str_ana if len(i) > 2] 
mx1=' '.join(str_ana1)
mx1=mx1.split() 

#function for counting most common words 
def vino_limpiar(xy):
    xy = [word for word in mx1 if word not in stopwords.words('english')]
    xy=[i for i in xy if len(i) > 1]
    xy=' '.join(xy)
    xy=re.findall('..............bordeaux..................',xy)
    xy=' '.join(xy)
    xy=xy.split()
    xy=[i for i in xy if len(i) > 2]
    no_words1=['san','warehouse','francisco','points','hollywood','points','inventory','wine','redwood','wines','city','read','finish','åèåê','main','robert','parker']
    xy = [word for word in xy if word not in no_words1]
    fruits_cnt=Counter(xy).most_common(50) 
    fruits=pd.DataFrame(fruits_cnt) 
    fruits.columns=['palabra','count']
    fruits=fruits.iloc[1:fruits.shape[0]]
    fruits=fruits[fruits['count']>=5]
    word="fruit"
    fruits = fruits[~fruits["palabra"].str.contains(word)]
    fruits['pct_palabra']=fruits['count']/(sum(fruits['count']))
    return fruits 
    
vino_limpiar(mx1)

#III. more text analysis 
mx2=mx1 
no_words=['san','warehouse','francisco','points','hollywood','points','inventory','wine','redwood',
'wines','city','read','finish','åèåê','main','robert','parker']
mx2 = [word for word in mx2 if word not in no_words]
tfidf=TfidfVectorizer()
features=tfidf.fit_transform(mx2)
tfidf_df=pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())
tops=tfidf_df.iloc[:,0:tfidf_df.shape[1]].mean() 
tops=tops.sort_values()
tops.to_csv("sa_tfidf.csv") 
count_words=Counter(mx2).most_common(30)  
count_words=pd.DataFrame(count_words)
count_words.columns=['word','count'] 
count_words['pct_word']=count_words['count']/(count_words['count'].sum())
count_words.to_csv("sa_count.csv") 
count_words 

#IV. Wine Ratings (points system)
cabs=fv1[fv1.pais=="Esp"]
cabs.shape
cabs=cabs[cabs['desc'].str.contains("cabernet sauvignon")] 
cabs=cabs['desc'].tolist()
lwww=' '.join(cabs)
lwww=lwww.split(' ')
lwww=[x.lower() for x in lwww]
lwww=' '.join(lwww)
mx3=re.findall('.....points',lwww)
points=pd.DataFrame(mx3)
points.columns=['points']
points['points'] = points['points'].str.replace(r"[a-zA-Z]",'')
points['points'] = points['points'].map(lambda x: x.lstrip('\r\r').rstrip('aAbBcC'))
points['points'] = points['points'].map(lambda x: x.lstrip('7/20').rstrip('aAbBcC'))
points['points'] =points['points'].map(lambda x: x.lstrip('6/20').rstrip('aAbBcC'))
points['points'] =points['points'].map(lambda x: x.lstrip('1-93').rstrip('aAbBcC'))
points['points'] =points['points'].map(lambda x: x.lstrip('-99').rstrip('aAbBcC'))
points['points']=points['points'].astype(float)
points['points'].mean() 

#V. text simlarities ************************************** 
pais1=fv1[fv1.pais=="Fra"]
word1="rhone"
word2="alsace"
pais1['reg1'] = np.where(pais1['desc'].str.contains(word1), 'yes', 'no')
pais1['reg2'] = np.where(pais1['desc'].str.contains(word2), 'yes', 'no')
reg1=pais1[pais1.reg1=="yes"] #region 1 
reg2=pais1[pais1.reg2=="yes"] #region 2 

#region 1 
listos_wines1=reg1['desc'].tolist()
listos_wines1=' '.join(listos_wines1)
listos_wines1=listos_wines1.split(' ')
listos_wines1=[x.lower() for x in listos_wines1]
listos_wines1 = list(filter(None, listos_wines1))
lwX=' '.join(listos_wines1)
lwX1=lwX.lower()
lwX1=re.sub(r'\W+', ' ', lwX1) 
list_ana1=lwX1.split(' ')
list_ana1=list(filter(None,list_ana1))
list_ana1 = [word for word in list_ana1 if word not in stopwords.words('english')]
list_ana1=[i for i in list_ana1 if len(i) > 1]
str_ana1=' '.join(list_ana1)
str_ana1=str_ana1.split()
str_ana1=[i for i in str_ana1 if len(i) > 2] 

#region 2 
listos_wines2=reg2['desc'].tolist()
listos_wines2=' '.join(listos_wines2)
listos_wines2=listos_wines2.split(' ')
listos_wines2=[x.lower() for x in listos_wines2]
listos_wines2 = list(filter(None, listos_wines2))
lwX2=' '.join(listos_wines2)
lwX2=lwX.lower()
lwX2=re.sub(r'\W+', ' ', lwX2) 
list_ana2=lwX2.split(' ')
list_ana2=list(filter(None,list_ana1))
list_ana2 = [word for word in list_ana2 if word not in stopwords.words('english')]
list_ana2=[i for i in list_ana1 if len(i) > 1]
str_ana2=' '.join(list_ana2)
str_ana2=str_ana2.split()
str_ana2=[i for i in str_ana2 if len(i) > 2] 

#region comparision 
match_wines=[i for i, j in zip(str_ana,str_ana1) if i==j]
s1=' '.join(str_ana)
s2=' '.join(str_ana1)

def similar(s1,s2):
    return SequenceMatcher(None,s1,s2).ratio()
similar(s1,s2)  

#VI. tfidf scores 
mx2=mx1 
no_words=['san','warehouse','francisco','points','hollywood','points','inventory','wine','redwood',
'wines']
mx2 = [word for word in mx2 if word not in no_words]
tfidf=TfidfVectorizer()
features=tfidf.fit_transform(mx2)
tfidf_df=pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())
tops=tfidf_df.iloc[:,0:tfidf_df.shape[1]].mean() 
tops.sort_values()

#VI. Wine Regions Comparisons 
rh=fra[fra.rhone=="yes"]
al=fra[fra.alsace=="yes"] 
listos_wines1=al['desc'].tolist()
listos_wines1=' '.join(listos_wines1)
listos_wines1=listos_wines1.split(' ')
listos_wines1=[x.lower() for x in listos_wines1]
listos_wines1 = list(filter(None, listos_wines1)
lwX=' '.join(listos_wines1)
lwX1=lwX.lower()
lwX1=re.sub(r'\W+', ' ', lwX1) 
list_ana1=lwX1.split(' ')
list_ana1=list(filter(None,list_ana1))

#word counting 
count_words1=Counter(list_ana1).most_common(50) 
count_words1 

#i. words before and after a specific wine word?
list_ana1 = [word for word in list_ana1 if word not in stopwords.words('english')]
list_ana1=[i for i in list_ana1 if len(i) > 1]
str_ana1=' '.join(list_ana1)
str_ana1=str_ana1.split()
str_ana1=[i for i in str_ana1 if len(i) > 2] 
match_wines=[i for i, j in zip(str_ana,str_ana1) if i==j] #matches words in the two lists 

s1=' '.join(str_ana)
s2=' '.join(str_ana1)

#similarity between two strings 
def similar(s1,s2):
    return SequenceMatcher(None,s1,s2).ratio()
similar(s1,s2) 

#VII. cosine similarity 
pais=fv1
pais=pais[pais.pais=="Esp"] #country 
listos_winesX=pais['desc'].tolist() 
listos_winesX=' '.join(listos_winesX)
listos_winesX=listos_winesX.split(' ')
listos_winesX=[x.lower() for x in listos_winesX]
listos_winesX = list(filter(None, listos_winesX))
lwXX=' '.join(listos_winesX)
lwX1=lwXX
lwX1=lwX1.lower()
lwX1=re.sub(r'\W+', ' ', lwX1) 
list_anaX=lwX1.split(' ')
list_anaX=list(filter(None,list_anaX))
list_anaX = [word for word in list_anaX if word not in stopwords.words('english')]
list_anaX=[i for i in list_anaX if len(i) > 1]
list_anaX=[x.lstrip('0123456789) ') for x in list_anaX]
list_anaX = list(filter(None, list_anaX)) 
str_ana=' '.join(list_anaX)
str_ana=str_ana.split()
str_ana1=[i for i in str_ana if len(i) > 2] 
mx1=' '.join(str_ana1) 
mx1=mx1.split() 
pais1=fv1
pais1=pais1[pais1.pais=="Fra"] #country 
listos_winesX=pais1['desc'].tolist() 
listos_winesX=' '.join(listos_winesX)
listos_winesX=listos_winesX.split(' ')
listos_winesX=[x.lower() for x in listos_winesX]
listos_winesX = list(filter(None, listos_winesX))
lwXX=' '.join(listos_winesX)
lwX1=lwXX
lwX1=lwX1.lower()
lwX1=re.sub(r'\W+', ' ', lwX1) 
list_anaX=lwX1.split(' ')
list_anaX=list(filter(None,list_anaX))

#words before and after a specific wine word 
list_anaX = [word for word in list_anaX if word not in stopwords.words('english')]
list_anaX=[i for i in list_anaX if len(i) > 1]
list_anaX=[x.lstrip('0123456789) ') for x in list_anaX]
list_anaX = list(filter(None, list_anaX)) 
str_ana=' '.join(list_anaX)
str_ana=str_ana.split()
str_ana1=[i for i in str_ana if len(i) > 2] 
#mx1=re.findall('....points',str_ana)
mx2=' '.join(str_ana1) #or mx1 
fra=mx2.split() 

#cosine similarity
WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


text1=' '.join(us) 
text2=' '.join(chile) 

vector1=text_to_vector(text1) 
vector2=text_to_vector(text2)

cosine=get_cosine(vector1,vector2) #cosine score

#**********************************************************************
#TF-IDF Analysis 
listos_winesX=[x.lower() for x in world] #world, paisX
listos_winesX = list(filter(None, listos_winesX))
lwXX=' '.join(listos_winesX)
lwX1=lwXX
lwX1=lwX1.lower()
lwX1=re.sub(r'\W+', ' ', lwX1) 

list_anaX=lwX1.split(' ')
list_anaX=list(filter(None,list_anaX))

#words before and after a specific wine word 
list_anaX = [word for word in list_anaX if word not in stopwords.words('english')]
list_anaX=[i for i in list_anaX if len(i) > 1]
list_anaX=[x.lstrip('0123456789) ') for x in list_anaX]
list_anaX = list(filter(None, list_anaX)) 
str_ana=' '.join(list_anaX)
str_ana=str_ana.split()
str_ana1=[i for i in str_ana if len(i) > 2] 
#mx1=re.findall('....points',str_ana)
mx1=' '.join(str_ana1)
mx1=mx1.split() 
mx2=mx1 
no_words=['san','warehouse','francisco','points','hollywood','points','inventory','wine','redwood',
'wines','city','read','finish','åèåê','main','robert','parker']
mx2 = [word for word in mx2 if word not in no_words]
tfidf=TfidfVectorizer()
features=tfidf.fit_transform(mx2)
tfidf_df=pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())
tops1=tfidf_df.iloc[:,0:tfidf_df.shape[1]].mean() 
tops1.to_csv("world_tfidf.csv")
final_tfidf=pd.merge(tops,tops1,on="word") #tops=us, tops1=world 
tops=tops.sort_values()

tfidf=TfidfVectorizer().fit_transform(mx1)
pairwise_sim=tfidf*tfidf.T 
pairwise_sim.todense()


no_words=['san','warehouse','francisco','points','hollywood','points','inventory','wine','redwood',
'wines'] 
tf1 = [word for word in mx1 if word not in no_words]
tfidf=TfidfVectorizer()
features=tfidf.fit_transform(mx1)
tfidf_df=pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())
tops=tfidf_df.iloc[:,0:tfidf_df.shape[1]].mean() 
tops=pd.DataFrame(tops)
tops.reset_index(level=0,inplace=True)
tops.columns=['word','tfidf_score']

l1=np.array(tops[['word','tfidf_score']])
l1=l1.tolist()


tf2 = [word for word in mx2 if word not in no_words]
tfidf=TfidfVectorizer()
features1=tfidf.fit_transform(mx2)
tfidf_df1=pd.DataFrame(features1.todense(),columns=tfidf.get_feature_names())
tops1=tfidf_df1.iloc[:,0:tfidf_df1.shape[1]].mean() 
tops1=pd.DataFrame(tops1)
tops1.reset_index(level=0,inplace=True)
tops1.columns=['word','tfidf_score']

l2=np.array(tops1[['word','tfidf_score']])
l2=l2.tolist()


fv1["price"] = pd.to_numeric(fv1.price, errors='coerce')
fv1["year"] = pd.to_numeric(fv1.year, errors='coerce')
fv1['desc']=fv1['desc'].astype(str)

non_null=fv1['desc'].notnull()

word="barolo"
owc_yes = fv1[fv1["desc"].str.contains(word,na=False)] 
owc_no = fv1[~fv1["desc"].str.contains(word)] 


fv1=pd.read_csv("fv1.csv",encoding="latin-1",lineterminator='\n')
fv1['desc'] = fv1['desc'].astype(str)
fv1['price'] = fv1['price'].apply(float)
fv1['year'] = fv1['year'].apply(float)
fv1=fv1[~fv1.pais.str.contains("JP")] 
fv1=fv1[~fv1.pais.str.contains("Eng")]
fv1=fv1[~fv1.pais.str.contains("Car")]
fv1=fv1[~fv1.pais.str.contains("Ire")]
fv1=fv1[~fv1.pais.str.contains("Sco")]
fv1=fv1[~fv1.pais.str.contains("Mex")]
fv1['desc']=fv1['desc'].str.lower() 

fv1[["price","year"]].describe() 
fv1.head(3)

pais=fv1[fv1.pais=="Esp"]

def word_avg(word):
   df=fv1 
   word_yes = df[df["desc"].str.contains(word)] 
   word_no = df[~df["desc"].str.contains(word)] 
   avg_yes=word_yes[["year","price"]].median()
   len_yes=len(word_yes)  
   avg_no=word_no[["year","price"]].median()
   return avg_yes,len_yes 

word_avg("zinfandel")   


def points_system(word):
   df=fv1[fv1.pais=="Chile"]
   vino_word=df[df["desc"].str.contains(word)] #word1, word2 (wines over x dollars?)
   vino_word=vino_word['desc'].tolist()
   vino_word=' '.join(vino_word)
   vino_word=vino_word.split(' ')
   vino_word=[x.lower() for x in vino_word]
   vino_word = list(filter(None, vino_word))
   vino_word=' '.join(vino_word)
   vino_word=vino_word.lower()
   vino_word=re.sub(r'\W+', ' ', vino_word) 
   vino_word=vino_word.split(' ')
   vino_word=list(filter(None,vino_word))
   vino_word=' '.join(vino_word)
   word_find=re.findall('...points',vino_word)
   word_strip=[s.strip('points') for s in word_find]
   word_strip=pd.DataFrame(word_strip)
   len_words=len(word_strip)
   word_strip.columns=['points']
   word_strip['points'] = word_strip['points'].str.extract('(\d+)', expand=False) 
   word_strip=word_strip['points'].astype(float)
   word_strip=pd.DataFrame(word_strip)
   avg_points=word_strip['points'].mean()
   return len_words, avg_points 

points_system("cabernet sauvignon")


def combos_wines(word1): #word2 
   w1 = fra[fra["desc"].str.contains(word1)] 
   len_combos=len(w1) 
   avg_combos=w1[["year","price"]].median()
   return len_combos,avg_combos

combos_wines("medoc")  


#count similar words?
def count_words(word1):
   df=fv1 
   df=df[df.pais==word1] 
   vino_word=df['desc'].tolist()
   vino_word=' '.join(vino_word)
   vino_word=vino_word.split(' ')
   vino_word=[x.lower() for x in vino_word]
   vino_word = list(filter(None, vino_word))
   vino_word=' '.join(vino_word)
   vino_word=vino_word.lower()
   vino_word=re.sub(r'\W+', ' ', vino_word) 
   vino_word=vino_word.split(' ')
   vino_word=list(filter(None,vino_word))
   vino_word=[word for word in vino_word if word not in stopwords.words('english')]
   no_words=['san','warehouse','francisco','points','hollywood','points','inventory','wine','redwood','wines','city','read','finish','åèåê','main','robert','parker','ã','â','points','parker','de','2018','2017','2016']
   vino_words = [word for word in vino_word if word not in no_words]
   count_words=Counter(vino_words).most_common(30)  
   count_words=pd.DataFrame(count_words)
   count_words.columns=['word','count'] 
   count_words['pct_word']=count_words['count']/(count_words['count'].sum())
   return count_words

count_words("Fra")

#most expensive wine words?
exp_wines=fv1[fv1.price>=70]
exp_wines.pais.unique()

cheap_wines=fv1[fv1.price<20]
pais=fv1[fv1.pais=="Ita"]

def wine_word_counter(df):
   sum_df=len(df)
   df=df['desc'].tolist() 
   xy = [word for word in df if word not in stopwords.words('english')]
   xy=[i for i in xy if len(i) > 1]
   xy=' '.join(xy)
   xy=xy.split()
   xy=[i for i in xy if len(i) > 2]
   no_words1=['the','all','with','palabra','more','san','warehouse','francisco','points','hollywood','points','inventory','wine','redwood','wines','city','read','finish','åèåê','main','robert','parker']
   xy = [word for word in xy if word not in no_words1]
   xy=[i for i in xy if len(i) > 1]
   xy=[x.lstrip('0123456789) ') for x in xy]
   xy = list(filter(None, xy))
   xy=' '.join(xy)
   xy=xy.split()
   xy=[i for i in xy if len(i) > 2] 
   xy=' '.join(xy)
   xy=xy.split() 
   fruits_cnt=Counter(xy).most_common(150) 
   fruits=pd.DataFrame(fruits_cnt) 
   fruits.columns=['palabra','count']
   fruits=fruits.iloc[1:fruits.shape[0]]
   fruits=fruits[fruits['count']>=5]
   fruits['pct_palabra']=fruits['count']/(sum_df)
   return fruits 

wine_word_counter(pais)













