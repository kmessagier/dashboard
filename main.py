import json
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.utils import parallel_backend
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular
import streamlit.components.v1 as components
import time
import requests
from urllib.request import urlopen


######################################################################

# création de la sidebar

#st.sidebar.title("MENU")
#st.sidebar.write('''
    ###  prédiction de défaut de paiement
 #   ''')
#st.sidebar.image('https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png', width=300)
#st.sidebar.image('images/finance-4858797_1280.jpg', width=300)


######################################################################





def main():
    API_URI = 'http://127.0.0.1:5000/api/client/'
# Création des containers

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# Le container header
with header:

    st.title('Crédit scoring')


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
######################################################################
# Le container dataset
with parallel_backend('threading', n_jobs=-1):
    with dataset:
        # chargement des données (data_test_poly) dans le cache


        if 'number_of_rows' not in st.session_state or 'type' not in st.session_state:
            st.session_state['number_of_rows'] = 5
            st.session_state['type'] = 'Categorical'


        first_data = load_data("data/data_test_50.csv")


        increment = st.button('Ajouter des lignes')
        if increment:
            st.session_state.number_of_rows += 1

        decrement = st.button('Supprimer des lignes')
        if decrement:
            st.session_state.number_of_rows -= 1

        st.dataframe(first_data.head(st.session_state['number_of_rows']))



    ##########################

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
            st.bar_chart(dist)

        else:
            st.table(first_data[column].describe())



        features1 = st.multiselect("Choisir la ou les colonnes: ", types['Numerical'], default=['AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'AMT_CREDIT'])

        st.write("You selected", len(features1), 'features')

        st.header("les graphiques")
        slider1 = st.slider('nombre de clients', min_value=5, max_value=50, value=25)

        if slider1:

            df = pd.DataFrame(first_data[:slider1], columns=features1)
            st.bar_chart(df.loc[:, features1])
            st.line_chart(df.loc[:, features1])

#######################################

if "id" not in st.session_state:
    st.session_state[id] = None
    st.sidebar.write('--------------------')
    st.markdown("Identifiant client")
    id = st.sidebar.selectbox("Identifiant client", first_data['SK_ID_CURR'], key='client')



features2 = st.sidebar.multiselect("Features2:", first_data.columns,
                                   default=['SK_ID_CURR', 'CODE_GENDER','DAYS_BIRTH_x', 'NAME_FAMILY_STATUS','CNT_CHILDREN', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL'])
st.write("You selected", len(features2), 'features')
st.sidebar.write('--------------------')


data_id = first_data.loc[first_data['SK_ID_CURR'] == id, features2]
data_id2 = data_id.astype('str')
st.dataframe(data_id2.T)




######################################################################


st.subheader('Boxplot des variables numériques')

features3 = st.sidebar.multiselect("Features3: ", types['Numerical'], default=['AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'AMT_CREDIT'])

st.write("You selected", len(features3), 'features')

st.sidebar.write('--------------------')



fig = px.box(first_data, y=features3,notched=True)
val = first_data.loc[first_data['SK_ID_CURR'] == id, features3].T

fig_id = px.bar(val, width=350, height=350)

col1, col2 = st.columns(2)
with col1:

    st.write(fig_id)

with col2:

    st.write(val)


st.write(fig)

st.sidebar.write('--------------------')
st.subheader('Boxplot numérique par variable catégorielle')

features4 = st.sidebar.multiselect("Features4 - Choisir une variable numérique: ", types['Numerical'],  default=['AMT_INCOME_TOTAL'])

st.write("You selected", len(features4), 'features')


category = st.sidebar.radio('choisir la variable catégorielle',
                            types['Categorical'])



fig3 = px.box(first_data, x= category, y=features4,
              color = category,
              points="outliers",
              notched=False,  # used notched shape

              )

st.write(fig3)



######################################################################




with st.form(key='modele LGBM'):



    if st.form_submit_button(label='Résultat de la procédure'):




        API_URI = 'https://credit-api-oc7.herokuapp.com/api/client/'
        response = urlopen(API_URI+str(id))

        data_json = json.loads(response.read())

        proba0 = float(data_json["proba0"])

        #json = data_json["json"]

        st.markdown(f'**SCORE CLIENT = {int(proba0*100)}/**100')




        discriminant = 0.4767

        #st.write(proba0, width=1000, height=300)
        if proba0 < discriminant:
            st.warning('**REFUS DE PRÊT**')
        else:
            st.success('**ACCORD DE PRÊT**')

        st.image('images/features_importance.jpg')



        #with parallel_backend('threading', n_jobs=6):
        #    y_pred_proba_1 = y_pred_proba[:50, 1]  # on ne garde les probabilités que pour les positifs (minoritaires)
        #    fpr, tpr, thresholds = roc_curve(y_test[:50], y_pred_proba_1)

            # Calcul du threshold seuil discriminant
        #    J = tpr - fpr
         #   ix = argmax(J)  # On va chercher la plus grande diff entre tpr et fpr pour avoir la valeur seuil
        #    best_thresh = thresholds[ix]
        #    st.write(f'Meilleur seuil discriminant {best_thresh}')

        # INTERPRETATION DU MODELE PAR LIME

        #pickle_in = open('modele/lgbm.pkl', 'rb')
        #lgbm = pickle.load(pickle_in)

        lgbm = load_models()

        data_client = load_data('data/mini_data_test.csv')

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress.progress(i+1)

        #st.dataframe(data_client.head(st.session_state['number_of_rows']))

        list_xtest = data_client.columns.tolist() # Création d'une liste récupérant les features
        #st.write(list_xtest)
        list_xtest_without_id = [e for e in list_xtest if e not in 'SK_ID_CURR'] #Modification de la liste sans SK_ID_CURR pour lancer la future prédiction
        #st.write(list_xtest_without_id)

        # Nouvelle donnée à interpréter
        X_try = data_client.iloc[:, :130].to_numpy()
        ix = data_client.index[data_client['SK_ID_CURR'] == id].tolist()[0]
        idx = X_try[ix, 0:130]

        #
        data_client_model = data_client.loc[:, list_xtest_without_id]

        explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(data_client.iloc[:50, :130]),
                                                      mode="classification",
                                                      feature_names=data_client_model.columns,
                                                      )

        explanation = explainer.explain_instance(idx, lgbm.predict_proba,)



        local_importance = explanation.as_pyplot_figure()
        #html_fig = exp.as_html( predict_proba=True, show_predicted_value=True,)

        # exp.show_in_notebook(show_table=True, )
        st.write(local_importance, )


       # html = explanation.as_html()

        #st.markdown(components.html(html, width=400, height=800))

        #shap_values = shap.TreeExplainer(lgbm).shap_values(data_client_model)
        #shapy = shap.summary_plot(shap_values, data_client_model, max_display=10)
        #st.write(shapy)

######################################################################
