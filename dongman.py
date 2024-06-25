import streamlit as st
import pandas as pd
import random
import torch
from fastai.collab import load_learner

learn = load_learner('best_model.pkl')

animes_df = pd.read_excel('anime.csv')
animes_df.columns = ['anime']
animes_df = animes_df.rename_azis('anime_id').reset_index()

def get_random_anime(n=3):
    return animes_df.sample(n).reset_index(drop=True)

def recommend_anime(user_ratings, n=5):
    user_id=max(user_ratings['user_id'])+ 1
    new_data = pd. DataFrame({'user_id': [user_id]*len(animes_df), 'anime_id': animes_df['anime_id'], 'anime': animes_df['anime']})
    dls = learn.dls.test_dl(new_data)
    preds, _= learn.get_preds(dl=dls)
    new_data['rating']= preds
    snew_data= new_data.sort_values(by='rating', ascending=False).head(n)
    return new_data.merge(animes_df, on='anime_id')

st.title('我们二次元是这样的')
st.header('动漫推荐')

if 'user_ratings' not in st.session_state:
    st.session_state['user_ratings']=pd.DataFrame(columns=['user_id','anime_id','rating'])
if 'random_animes' not in st.session_state:
    st.session_state:['random_animes'] = get_random_animes()

st.subheader('请对动漫进行评分(1-5分)')
user_id = 1
for i, joke in st.session_state['random_animes'].iterrows():
    rating = st.slider(f'动漫{i+1}: {anime["anime"]}', 1,5, key=f'rating_{i}')
    if st.button(f'提交评分 动漫{i+1}', key=f'button_{i}'):
        new_rating = pd.DataFrame({'user_id': [user_id], 'anime_id': [anime['anime_id']], 'rating': [rating]})
        st.session_state['user_ratings'] = pd.concat([st.session_state['user_ratings'], new_rating], ignore_index=True)
        st.write('评分已提交! ')

if st.button('推荐动漫'):
    recommended_animes = recommended_animes(st.session_state['user_ratings'])
    st.session_state['recommended_animes'] = recommended_animes


if 'recommended_animes' in st.session_state:
    st.subheader('推荐动漫')
    total_rating = 0
    for i. anime in st.session_state['recommended_animes'].iterrows():
        rating = st.slider(f'推荐动漫[i+1]: {anime["anime"]}',1,5,key=f'rec_rating_{1}')
        total_rating +=rating
    if len(st.session_state['recommended_animes']) >0:
        satisfaction = total_rating/len(st.session_state['recommended_animes'])
        st.write(f'用户满意度：{satisfaction:.2f} / 5')