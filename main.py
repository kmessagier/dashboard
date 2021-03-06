import json
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.utils import parallel_backend
import numpy as np
from lime import lime_tabular
import streamlit.components.v1 as components
import time
import requests
from urllib.request import urlopen
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
######################################################################

# création de la sidebar

#st.sidebar.title("MENU")
#st.sidebar.write('''
    ###  prédiction de défaut de paiement
 #   ''')
#st.sidebar.image('https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png', width=300)
#st.sidebar.image('images/finance-4858797_1280.jpg', width=300)


######################################################################


# Use the full page instead of a narrow central column
#st.set_page_config(layout="wide")



def main():
    API_URI = 'https://credit-api-oc7.herokuapp.com/api/client/'

# Création des containers

header = st.container()
dataset = st.container()
graphes = st.container()
model_training = st.container()

# Le container header
with header:

    st.title('APPLICATION de CREDIT SCORING')



# Fonction permettant de charger les données
@st.cache(allow_output_mutation=True)
def load_data(filename):
    data = pd.read_csv(filename)
    data.drop(columns=['Unnamed: 0'], inplace=True)

    return data

# loading the trained model
@st.cache(persist=True)
def load_models():
    with open('modele/lgbm.pkl', 'rb') as file:
        lgbm = pickle.load(file)

    return lgbm

def resultat():
    API_URI = 'https://credit-api-oc7.herokuapp.com/api/client/'
    response = urlopen(API_URI+str(id))

    data_json = json.loads(response.read())

    proba0 = float(data_json["proba0"])
    score = int(proba0*100)
    #json = data_json["json"]

    discriminant = 0.485
    #st.write(proba0, width=1000, height=300)
    if proba0 < discriminant:
        st.sidebar.warning('**REFUS DE PRÊT**')
    else:
        st.sidebar.success('**ACCORD DE PRÊT**')


    indicateur = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=score,
        mode="gauge+number+delta",
        title={'text':'Score'},
        delta={'reference': discriminant*100, 'increasing': {'color': "RebeccaPurple"}},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgray"},
                   {'range': [50, 100], 'color': "cyan"}],
               'threshold': {'line': {'color': "orange", 'width': 4}, 'thickness': 0.75, 'value':discriminant*100 }}))

    indicateur.update_layout(
        width=270,
        height=270,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
    )
    )
    st.sidebar.write(indicateur)


#####################-------ANALYSE DONNEES------###############################
my_expander_A = st.expander(label= 'Analyse globale', expanded=True)
with my_expander_A:



    # chargement des données (data_test_poly) dans le cache


    if 'number_of_rows' not in st.session_state or 'type' not in st.session_state:
        st.session_state['number_of_rows'] = 2
        st.session_state['type'] = 'Categorical'


    first_data = load_data("data/data_test_50.csv")
    first_data['SK_ID_CURR'] = first_data['SK_ID_CURR'].astype('int')


    increment = st.button('Ajouter des lignes')
    if increment:
        st.session_state.number_of_rows += 1

    decrement = st.button('Supprimer des lignes')
    if decrement:
        st.session_state.number_of_rows -= 1

    st.dataframe(first_data.head(st.session_state['number_of_rows']))



##########################-----DESCRIBE et graphe-----#####################

    types = {'Numerical':
                 first_data.select_dtypes(include=np.number).columns.tolist(),
             'Categorical' :
                 first_data.select_dtypes(exclude=np.number).columns.tolist()}

    column = st.selectbox('Choisir une colonne', types[st.session_state['type']])

    def handle_click_without_button():
        if st.session_state.kind_of_column:
            st.session_state.type = st.session_state.kind_of_column

    type_of_column = st.radio("Quel type d'analyse souhaitez-vous?",
                              ['Categorical', 'Numerical'],
                              on_change=handle_click_without_button,
                              key='kind_of_column')


    if st.session_state['type'] ==  'Categorical':
        dist = pd.DataFrame(first_data[column].value_counts()).head(15)
        fig0 = px.bar(dist, x=dist.index, y =column,
                      width=700, height=400,
                      title=f'Diagramme en barres de {column}')
        fig0.update_layout(plot_bgcolor="white")
        st.write(fig0)

    else:
        st.table(first_data[column].describe())



    #####################-------IDENTIFIANT-sidebar------###############################

    if "id" not in st.session_state:
        st.session_state[id] = None
        st.sidebar.write('--------------------')
        id = st.sidebar.selectbox('CLIENT ID',first_data['SK_ID_CURR'], key='client')

        #################---RESULTAT---###############################

        resultat()




    #################---GRAPHIQUES---##############################################

###################----Graphique-1----##################
my_expander1 = st.expander(label='Graphique 1', expanded=True)
with my_expander1:
    st.sidebar.markdown('**GRAPHIQUE 1**')

    features1 = st.sidebar.multiselect("Une ou plusieurs colonnes: ", types['Numerical'], default=['AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'AMT_CREDIT'])

    st.write("Vous avez sélectionné", len(features1), 'variables')


    slider1 = st.slider('nombre de clients', min_value=1, max_value=50, value=25)

    if slider1:

        df = pd.DataFrame(first_data[:slider1], columns=features1)
        separation = ',  '.join(features1)
        fig1 = px.bar(first_data[:slider1],
                      y=features1,
                      width=800,
                      title = f'Graphique en bâtons de {separation}',
                      color_discrete_sequence= px.colors.qualitative.G10)
        fig1.update_layout(plot_bgcolor="white")


        st.write(fig1)
        separation = ',  '.join(features1)
        fig12 = px.line(df.loc[:, features1], width=800,
                        color_discrete_sequence=px.colors.qualitative.G10,
                        title = f'Courbe de {separation}')
        fig12.update_layout(plot_bgcolor="white")
        fig12.update_xaxes(showgrid=True,linewidth=1, linecolor='gray', mirror=True, gridcolor="Lightgray")
        fig12.update_yaxes(showgrid=True,linewidth=1, linecolor='gray', mirror=True, gridcolor="Lightgray")

        st.write(fig12)




###################----Graphique-2----##################

my_expander = st.expander(label='Graphique 2',expanded=True)
with my_expander:


    st.sidebar.markdown('**GRAPHIQUE 2**')

    features4 = st.sidebar.multiselect("une variable numérique: ", types['Numerical'],  default=['AMT_INCOME_TOTAL'])

    category = st.radio('une variable catégorielle',
                        types['Categorical'],6)

    try:
        fig2 = px.box(first_data, x= features4, y=category,
                      color = category,
                      points="outliers",
                      notched=False,  # used notched shape
                      width=800,
                      height=500,
                      orientation='h',
                      title=f'Boîte à moustache de {features4[0]} éclatée sur la variable catégorielle {category}'
                      )
        fig2.update_layout(plot_bgcolor="white")
        fig2.update_xaxes(showgrid=False,linewidth=1, linecolor='gray', mirror=True, )
        fig2.update_yaxes(showgrid=False,linewidth=1, linecolor='gray', mirror=True, )


        st.write(fig2)

    except IndexError:
        st.warning('ERREUR - Veuillez saisir une  variable dans la sidebar GRAPHIQUE 5')

################GRAPHIQUE 3 et 4##########################

my_expander = st.expander(label='Graphiques 3 et 4', expanded=True)
with my_expander:

    st.sidebar.markdown('**GRAPHIQUES 3 et 4**')
    features3 = st.sidebar.multiselect("Sélectionner les variables numériques: ", types['Numerical'], default=['AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'AMT_CREDIT'])

    st.write("Vous avez sélectionné", len(features3), 'variable(s)')
    st.sidebar.write('--------------------')



    separation = ',  '.join(features3)
    fig3 = px.box(first_data, y=features3,notched=True,
                  title=f'Graphique à moustache de {separation}',
                  )
    fig3.update_layout(plot_bgcolor="white")
    fig3.update_xaxes(showgrid=False, linewidth=1, linecolor='gray', mirror=True, )
    fig3.update_yaxes(showgrid=False, linewidth=1, linecolor='gray', mirror=True, )
    val = first_data.loc[first_data['SK_ID_CURR'] == id, features3].T

    fig4 = px.bar(val, width=400, height=300)
    fig4.update_layout(plot_bgcolor="white")
    fig4.update_xaxes(showgrid=False, linewidth=1, linecolor='gray', mirror=True, )
    fig4.update_yaxes(showgrid=False, linewidth=1, linecolor='gray', mirror=True, )
    st.write(fig3)

    st.write(fig4)

 #####################-------TABLEAU------###############################


my_expander0 = st.expander(label='Analyse client',expanded=True)
with my_expander0:
    st.sidebar.markdown('**Données**')
    features2 = st.sidebar.multiselect("les variables:", first_data.columns,
                                       default=['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH_x',
                                                'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'CREDIT_TERM',
                                                'DAYS_ID_PUBLISH', 'AMT_INCOME_TOTAL',
                                                'AMT_CREDIT',
                                                'AMT_ANNUITY', 'DAYS_EMPLOYED'])

    st.sidebar.write('--------------------')

    st.markdown('**Données importantes du client**')
    st.write("Vous avez sélectionné", len(features2), 'variables')

    data_id = first_data.loc[first_data['SK_ID_CURR'] == id, features2]
    data_id2 = data_id.astype('str')
    st.table(data_id2.T)

###################----Graphique-5----##################

st.sidebar.markdown('**GRAPHIQUE 5**')

my_expander = st.expander(label='Graphique 5', expanded=True)
with my_expander:
    features_dbl_1 = st.sidebar.multiselect("Choisir 2 variables ",
                                            first_data.columns, default=['AMT_CREDIT', 'AMT_INCOME_TOTAL'])

    category_2 = st.radio('', types['Categorical'], 7)

    slider2 = st.slider('Nombre de clients', min_value=1, max_value=50, value=50)

    df1 = first_data[features_dbl_1]
    df_slider2 = df1[:slider2]

    try:
        fig5 = px.scatter(df_slider2, x=features_dbl_1[0], y=features_dbl_1[1],
                          width=800,
                          height=450,
                          color=first_data.loc[:slider2 - 1, category_2],
                          opacity=1,
                          title=f'Graphique de {first_data[features_dbl_1].columns[0]} en fonction de {first_data[features_dbl_1].columns[1]} distribué sur {category_2}',
                          )
        fig5.update_layout(plot_bgcolor="white",)
        did = first_data.loc[first_data['SK_ID_CURR'] == id, df1.columns,]
        fig5.update_xaxes(showgrid=True,linewidth=1, linecolor='gray', mirror=True, gridcolor="Lightgray")
        fig5.update_yaxes(showgrid=True,linewidth=1, linecolor='gray', mirror=True, gridcolor="Lightgray")
        fig5.add_trace(
            go.Scatter(
                x=[did.iloc[0, 0]],
                y=[did.iloc[0, 1]],
                mode="markers+text",

                marker=dict(symbol= "star", size=13,
                            color="Lightskyblue",
                            line_color="midnightblue",
                            line_width=2),
                textposition="middle right",
                text = f"Client n° {id}",
                showlegend=False)
        )


        # fig2.update_layout(px.scatter(did.iloc[0, 0], did.iloc[0, 1],symbol='x', color= 'black' )
        # plt.title(
        #    f'Graphique de {first_data[features_dbl_1].columns[0]} en fonction de {first_data[features_dbl_1].columns[1]} éclaté sur {category_2}')
        st.write(first_data.loc[first_data['SK_ID_CURR'] == id, [category_2]])
        st.write(fig5)
    except IndexError:
        st.warning('ERREUR - Veuillez saisir 2 variables variable dans la sidebar GRAPHIQUE2')

    st.sidebar.write('--------------------')

#########################"--------EXPLICATIONS-GLOBALE-------############################
#@st.cache(persist=True)
my_expander = st.expander(label='Interprétation',expanded=True)
with my_expander:
    with st.form(key='modele LGBM'):
        if st.form_submit_button(label='Interprétation globale et locale'):


            st.image('images/Features_importance.jpg')





            ################################--_EXPLICATIONS-LOCALE---#################################
            lgbm = load_models()

            data_client = load_data('data/mini_data_test.csv')


            #st.dataframe(data_client.head(st.session_state['number_of_rows']))

            list_xtest = data_client.columns.tolist() # Création d'une liste récupérant les features
            #st.write(list_xtest)
            list_xtest_without_id = [e for e in list_xtest if e not in 'SK_ID_CURR'] #Modification de la liste sans SK_ID_CURR pour lancer la future prédiction
            #st.write(list_xtest_without_id)

            # Nouvelle donnée à interpréter
            X_try = data_client.iloc[:, :130].to_numpy()
            ix = data_client.index[data_client['SK_ID_CURR'] == id].tolist()[0]
            idx = X_try[ix, :]


            #
            data_client_model = data_client.loc[:, list_xtest_without_id]

            explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(data_client.iloc[:, :130]),

                                                          feature_names=data_client_model.columns,
                                                          )

            explanation = explainer.explain_instance(idx, lgbm.predict_proba,)



            local_importance = explanation.as_pyplot_figure()



            st.write(local_importance, )


            #html = explanation.as_html()


            #st.markdown(components.html(html, width=800, height=800))


######################################################################

if __name__ == '__main__':
    main()
