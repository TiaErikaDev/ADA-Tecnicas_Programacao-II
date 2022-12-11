import streamlit as st
import plotly.express as px 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

st.title('Dataset Copa do Mundo - Visualização de dados')

st.header('Trabalho desenvolvido na disciplina de técnicas de programação II com a orientação do professor Thiago Medeiros.')
st.text('https://www.linkedin.com/in/tiaerikadev/')

st.text('Visualizando a base de dados da copa do mundo')
df = pd.read_csv('https://raw.githubusercontent.com/mdrs-thiago/uerj-topicos-a/main/datasets/WorldCupMatches.csv')
st.dataframe(df)


col1, col2, col3 = st.columns(3)
df = pd.read_csv('https://raw.githubusercontent.com/mdrs-thiago/uerj-topicos-a/main/datasets/WorldCupMatches.csv')

#Retiramos duplicatas.
df.drop_duplicates(inplace=True)

#Precisamos também tirar a última linha.
df = df.iloc[:-1,:]

rn_list = df[df['Home Team Name'].str.contains('>')]['Home Team Name'].unique().tolist()
rn_list_away = df[df['Away Team Name'].str.contains('>')]['Away Team Name'].unique().tolist()

rn_list.extend(rn_list_away)

#Pegando os valores da lista não-duplicados
wrong = list(set(rn_list))

right = [n.split('>')[-1] for n in wrong]

#Outros nomes que faremos.
wrong.extend(['Germany FR', 'IR Iran'])

right.extend(['Germany', 'Iran'])

#Dict comprehension
dict_changes = {k:v for k,v in zip(wrong, right)}

df['Home Team Name'].replace(dict_changes,inplace=True)
df['Away Team Name'].replace(dict_changes,inplace=True)

#Vamos criar uma coluna auxiliar para contar os vencedores.
df['Winner'] = np.nan

def get_winner(row):
    if row['Home Team Goals'] > row['Away Team Goals']:
        row['Winner'] = row['Home Team Name']
    elif row['Home Team Goals'] < row['Away Team Goals']:
        row['Winner'] = row['Away Team Name']
    else:
        if (row['Win conditions'] == ' ') or (row['Win conditions'] is None):
            row['Winner'] = 'Tie'
        else:
            row['Winner'] = row['Win conditions'].split(' win')[0]
    return row

df = df.apply(get_winner, axis=1)

#Manualmente adicionando.
new_vals = ['Germany','Argentina','Argentina','Germany']

df.loc[df['Winner'] == '','Winner'] = new_vals

titles = df[df['Stage'] == 'Final']['Winner'].value_counts()

#Criando o gráfico
fig = plt.figure(figsize=(16,10))
colors = ['Yellow','Green','Red','White','lightblue','White','darkblue','darkred']
edge_color = ['Green','Red','Black','Blue','Blue','Red','Red','Yellow']

plt.bar(titles.index, titles.values, color=colors, edgecolor=edge_color, alpha=0.8, linewidth=3)
plt.grid(axis='y')
plt.xticks(fontsize=15, rotation = 45);
plt.xlabel('Países',fontsize=12)
plt.yticks(fontsize=15);
plt.ylabel('Títulos',fontsize=12)
plt.title('Campeões da Copa do Mundo',fontsize=20)

st.pyplot(fig)


#criando o gráfico 1/3
df_home = df.groupby(['Year', 'Home Team Name']).count()['Home Team Goals'].to_frame()
df_away = df.groupby(['Year', 'Away Team Name']).count()['Away Team Goals'].to_frame()

df_home = df_home.reset_index().rename(columns={'Home Team Name': 'Name', 'Home Team Goals':'Home Matches'}).set_index(['Year','Name'])
df_away = df_away.reset_index().rename(columns={'Away Team Name': 'Name', 'Away Team Goals':'Away Matches'}).set_index(['Year','Name'])

df_concat = pd.concat([df_home, df_away], axis='columns')

df_concat.fillna(0,inplace=True)

df_concat['Total Matches'] = df_concat['Home Matches'] + df_concat['Away Matches']

#Pegando os países com maiores participações.
df_concat.reset_index()['Name'].value_counts()

part_teams = df_concat.reset_index()['Name'].value_counts()[:10]

with col1:
    st.text('Participações em Copas do Mundo')
    
    fig = plt.figure(figsize=(16,10))
    plt.barh(part_teams.index, part_teams.values, color='orange', edgecolor='black', linewidth=4)

    xticks = np.arange(0, 21, 2)
    plt.grid(axis='x')
    plt.xticks(xticks, fontsize=15);
    plt.xlabel('Participações',fontsize=18)
    plt.yticks(fontsize=15);
    plt.ylabel('Países',fontsize=18)
    #plt.title('Participações em Copa do Mundo',fontsize=20)

    st.pyplot(fig)
    
    
#criando o gráfico 2/3
plays = df_concat.reset_index()
total_plays = plays.groupby('Name').agg({'Total Matches': 'sum'}).sort_values(by='Total Matches', ascending=False)
top_10 = total_plays.iloc[:10,:]

with col2:
    st.text('Partidas em Copas do Mundo')
    fig = plt.figure(figsize=(16,10))
    plt.barh(top_10.index, top_10['Total Matches'], color='orange', edgecolor='black', linewidth=3)

    xticks = np.arange(0, 121, 20)
    plt.grid(axis='x')
    plt.xticks(xticks, fontsize=15);
    plt.xlabel('Participações',fontsize=18)
    plt.yticks(fontsize=15);
    plt.ylabel('Países',fontsize=18)
    #plt.title('Participações em Copa do Mundo',fontsize=20)
    st.pyplot(fig)    
    
    
#criando o gráfico 2/3
home_goals = df.groupby(['Year', 'Home Team Name']).agg({'Home Team Goals':'sum'})
away_goals = df.groupby(['Year', 'Away Team Name']).agg({'Away Team Goals':'sum'})

home_goals = home_goals.reset_index().rename(columns={'Home Team Name': 'Name'}).set_index(['Year','Name'])
away_goals = away_goals.reset_index().rename(columns={'Away Team Name': 'Name'}).set_index(['Year','Name'])

total_goals = pd.concat([home_goals, away_goals], axis='columns')

total_goals.fillna(0,inplace=True)

total_goals['Total Goals'] = total_goals['Home Team Goals'] + total_goals['Away Team Goals']

goals = total_goals.reset_index()
team_goals = goals.groupby('Name').agg({'Total Goals':'sum'}).sort_values(by='Total Goals',ascending=False)


top_10_scores = team_goals.iloc[:10,:]
with col3:
    st.text('Gols em Copas do Mundo')
    fig = plt.figure(figsize=(16,10))
    plt.barh(top_10_scores.index, top_10_scores['Total Goals'], color='orange', edgecolor='black', linewidth=3)

    xticks = np.arange(0, 250, 20)
    plt.grid(axis='x')
    plt.xticks(xticks, fontsize=15);
    plt.xlabel('Participações',fontsize=18)
    plt.yticks(fontsize=15);
    plt.ylabel('Países',fontsize=18)
    #plt.title('Participações em Copa do Mundo',fontsize=20)
    st.pyplot(fig)
    

col21, col22 = st.columns(2)

with col21:
    country_select = st.selectbox('Escolha um país para realizar o retrospecto em Copa do Mundo.',['Brazil','France','Germany','Argentina','Italy'])

    df_brazil = df[(df['Home Team Name'] == country_select) | (df['Away Team Name'] == country_select)]

    df_brazil.loc[df_brazil['Winner'] == country_select,'Winner'] = 'Win'

    df_brazil.loc[~df_brazil['Winner'].isin(['Win','Tie']),'Winner'] = 'Lose'

    recap = df_brazil['Winner'].value_counts()

with col22:
    fig = plt.figure(figsize=(16,16))
    colors = ['green','red','gray']

    wedgeprops={"edgecolor":"white",'linewidth': 2, 'antialiased': True}

    plt.pie(recap.values, autopct='%.2f%%', colors=colors, wedgeprops=wedgeprops,textprops={'fontsize': 18, 'color':'black'});

    plt.legend(recap.index, prop={'size':25}, loc='upper right')

    st.pyplot(fig)
    
    
espect = df.groupby('Year').agg({'Attendance':'sum'})

espect.loc[1942,'Attendance'] = 0
espect.loc[1946,'Attendance'] = 0

espect = espect.sort_index()

list_hosts = ['Uruguay','Italy','NA','NA','France','Brazil','Switzerland','Sweden','Chile','England','Mexico','Germany','Argentina','Spain','Mexico','Italy','United States','France','Japan/Korea','Germany','South Africa','Brazil']


fig = plt.figure(figsize=(32,15))
#st.text('Média de público nos estádios por copa do mundo')
plt.plot(espect.index, espect['Attendance'], linewidth=5)
plt.plot(espect.index, espect['Attendance'], 'o', markersize=10)

yvals = np.arange(500000, 4000000, 500000)
plt.yticks(yvals, ['500K','1M', '1.5M', '2M', '2.5M', '3M', '3.5M'], fontsize=18);
xvals = np.arange(1930, 2018, 4)

xalias = [hosts + ', ' + str(edition) for hosts,edition in zip(list_hosts, xvals)]

plt.xticks(xvals, xalias, rotation=60, fontsize=18);
plt.grid(linewidth=2)
plt.ylim([0,4000000])
plt.title('Média de público nos estádios por copa do mundo',fontsize=40);

st.pyplot(fig)


col31, col32 = st.columns(2)

with col31:
    country = st.selectbox('Escolha outro país para realizar o retrospecto em Copa do Mundo.',['Brazil','France','Germany','Argentina'])

    df_brazil = df[(df['Home Team Name'] == country) | (df['Away Team Name'] == country)]

    df_brazil.loc[df_brazil['Winner'] == country,'Winner'] = 'Win'

    df_brazil.loc[~df_brazil['Winner'].isin(['Win','Tie']),'Winner'] = 'Lose'

    recap = df_brazil['Winner'].value_counts().to_frame().reset_index()
    recap.rename(columns={'index':'Win?', 'Winner':'Num Matches'},inplace=True)

with col32:
    fig = plt.figure(figsize=(16,16))
    colors = ['green','red','gray']

    wedgeprops={"edgecolor":"white",'linewidth': 2, 'antialiased': True}

    fig = px.pie(recap, names='Win?', values='Num Matches', title=f'Retrospecto de {country}')
    st.write(fig)


def apply_espectators(df):
    attendance = df['Attendance'].sum()
    average  = df['Attendance'].mean()
    if np.any(df['Stage'] == 'Final'):
        winner = df[df['Stage'] == 'Final']['Winner'].values[0]
    else:
        winner = 'No Winner'

    new_df = pd.DataFrame([{'Attendance':attendance, 'Avg Attendance': average, 'Winner':winner}])
    print(new_df)
    return new_df

