#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.head()


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


movies.head()


# In[7]:


movies.shape


# In[8]:


movies.info()


# In[9]:


credits.info()


# In[10]:


movies['crew'].count


# In[11]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[12]:


movies.head(2)


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.iloc[0].genres


# In[15]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[16]:


import ast


# In[17]:


movies['genres']=movies['genres'].apply(convert)


# In[18]:


movies['keywords']=movies['keywords'].apply(convert)


# In[19]:


movies.head()


# In[ ]:





# In[ ]:





# In[20]:


movies.head()


# In[21]:


def convert1(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[22]:


movies['cast']=movies['cast'].apply(convert1)


# In[23]:


def convert2(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[24]:


movies['crew']=movies['crew'].apply(convert2)


# 

# movie.head()

# In[25]:


movies.head()


# In[26]:


movies['overview'][0]


# In[27]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# mo

# In[28]:


movies.head()


# In[29]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])


# In[30]:


movies.head()


# In[31]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[79]:


movies.head()


# In[145]:


movies['title']


# In[99]:


new_df=movies[['movie_id','title','tags']]


# In[100]:


new_df.head()


# In[101]:


import nltk


# In[102]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[103]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return ' '.join(y)


# In[104]:


new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))


# In[105]:


new_df['tags'][0]


# In[106]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[107]:


new_df['tags'][0]


# In[ ]:





# In[108]:


new_df.head()


# In[109]:


new_df.shape


# In[110]:


new_df['tags']=new_df['tags'].apply(stem)


# In[112]:


new_df.head()


# In[113]:


from sklearn.feature_extraction.text import CountVectorizer


# In[114]:


cv=CountVectorizer(max_features=5000,stop_words='english')


# In[115]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[116]:


vectors


# In[117]:


cv.get_feature_names()


# In[62]:


ps.stem('loving')


# In[146]:


from sklearn.metrics.pairwise import cosine_similarity


# In[121]:


similarity=cosine_similarity(vectors)


# In[136]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[141]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
#        print(i[0])


# In[143]:


recommend('Aliens')


# In[140]:


new_df.iloc[1216].title


# In[127]:


new_df[new_df['title']=='Batman Begins'].index[0]


# In[132]:


a=new_df['title']=='Batman Begins'
new_df[a].index[0]


# In[ ]:




