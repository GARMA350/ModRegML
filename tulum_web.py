import streamlit as st
import pickle
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns


# Cargar modelos

scaler = load('scaler.joblib')
OLS = load('modelo_lineal.joblib')
SVR = load('modelo_SVR.joblib')
RFR = load('modelo_RFR.joblib')
XGB = load('modelo_XGB.joblib')


# Título de la aplicación
st.markdown('<h1><center>Modelos de Regresión con Machine Learning</center></h1>',unsafe_allow_html=True)
st.write("<h3><center>Luis Armando García Rodríguez (GARMA)</center><h2>",unsafe_allow_html=True)

st.write("""
         <style>
           p {
            text-align: justify;
             }
         </style>
         <h4>Presentación</h4>
         <p>El Machine Learning ha revolucionado la manera en que analizamos, modelamos e interpretamos datos en la actualidad. Este concepto no es nuevo, pues tiene inicios desde la decada de los 50's, sin embargo fue hasta la decada de 2010 donde esté empezo a evolucionar exponencialmente. Dicho progreso se debe en gran medida, al incremento en la capacidad computacional de la que disponemos hoy en dia, que nos ha permitido explorar nuevas metodologías para el modelado de datos. Estas han conseguido innovaciones y mejoras significativas en comparación con los modelos tradicionales que ya conocíamos.<p>
         <p>Un ejemplo de este avance son los Modelos de Regresión basados en Machine Learning, los cuales han demostrado una capacidad de inferencia superior que la de su antecesor directo, la regresión lineal. Estos modelos de regresión con ML son capaces capturar relaciones no lineales entre las variables, algo que la regresión, con el enfoque de linealidad, no puede abordar eficazmente.<p>
         <br>
         <p>En esta aplicación web, he desarrollado algunos de los modelos de regresión de ML para poder compararlos entre sí y con la regresión lineal. El objetivo de este proyecto es demostrar la capacidad y superioridad de los modelos de ML cuando las relaciones en los datos son complejas y no lineales. Para ello, utilizaremos un caso práctico con datos reales y complejos, donde vamos a estimar el precio por noche de un alquiler en Tulum (México) en función de los atributos con los que cuenta dicho alquiler. El conjunto de datos a utilizar contiene el precio de 1019 alquileres anunciados en la plataforma Booking.com, disponibles al 11 de febrero de 2021. Dicho conjunto fue recuperado del libro "Introducción a la valorización económica ambiental: teoría y práctica", escrito por Karina Caballero Güendulain y Saúl Basurto Hernández.<p>
          """,unsafe_allow_html=True)

st.markdown("""
            <p><h4>Análisis Exploratorio de Datos</h4><p>
            <p>Antes de modelar nuestros datos, es importante conocer su estructura, el tipo de variables con las que vamos a trabajar, conocer los estadísticos descriptivos principales e identificar las relaciones entre variables. Para ello, se realiza el análisis exploratorio de datos, el cual, en este caso, inicia con una presentación de los estadísticos descriptivos más relevantes por variable. Esto nos proporcionará mayor información para identificar las técnicas de modelado que debemos aplicar.<p>
            <p><strong>Estadísticos Descriptivos</strong>
            """,unsafe_allow_html=True)

#Estadisticas descriptivas
descriptivas = pd.read_csv("descrip_tulum.csv")
st.write(descriptivas)

st.markdown("""
            <p>Como puede apreciarse en la tabla anterior, no contamos con valores nulos (N/A), por lo que no es necesario hacer ningún tipo de imputación de los mismos. Podemos identificar que contamos con 14 variables, de las cuales 1 es continua (precio), 2 son numéricas discretas (superficie y capacidad), y las otras 11 son dicotómicas (vista al mar, aire acondicionado, wifi, balcón, terraza, pueblo, TV, alberca, estacionamiento, gimnasio y bar), las cuales toman el valor de 1, cuando el atributo es verdadero (es decir, que el alquiler sí cuenta con ese atributo) y toman el valor 0, cuando no lo es (cuando el alquiler no cuenta con ese atributo). Si nos fijamos en las variables (precio y superficie), más específicamente en las estadísticas de su distribución univariante (Min, 1Q, 2Q, 3Q, Max), podemos darnos cuenta de que ambas distribuciones están sumamente sesgadas, al igual que su media y desviación estándar, debido a que estos son indicadores sensibles a outliers, que evidentemente están presentes en la distribución. Para poder visualizar esto de mejor manera, vamos a recurrir a graficar boxplots, también conocidos como diagramas de cajas y bigotes.</p>
            <p><strong>Boxplots</strong><p>
            """,unsafe_allow_html=True)


#Bases

Base = pd.read_csv('Base_Tulum.csv')
Base_T1 = Base[['precio', 'vistaalmar', 'superficie','terraza']].copy()
Base_T2 = Base[['precio', 'pueblo', 'tv', 'alberca']].copy()
Base_T3 = Base[['precio', 'estacionamiento', 'gimnasio','bar']].copy()


# Boxplot precio
plt.figure(figsize=(8, 2))
sns.boxplot(Base[["precio"]],orient="h",color="#49007e")
plt.title('Boxplot de Precio')
plt.show()
st.pyplot(plt)

#Boxplot superficie
plt.figure(figsize=(8, 2))
sns.boxplot(Base[["superficie"]],orient="h",color="#ff7d10")
plt.title('Boxplot de Superficie')
plt.show()
st.pyplot(plt)

st.markdown("""
          <p>Al visualizar los boxplots, notamos que estamos frente a dos distribuciones grandemente sesgadas (como ya lo afirmábamos), pues vemos superficies y precios muy por encima de 1.5 veces el rango intercuartílico (Q3-Q1), que posiblemente estén mutuamente asociados. Esto nos habla de las extensas cantidades de terreno y los exorbitantes precios que podemos encontrar dentro de los alquileres, sin embargo, esto no es un problema en sí para el modelaje. Estos outliers están en el espacio univariante y pueden ser o no outliers en el espacio multivariante. En caso de no serlo, esto no traería ningún tipo de problema al modelaje, pues nuestro modelo será capaz de realizar una buena estimación en dicho caso. Sin embargo, si resultan ser outliers en el espacio multivariante, esto sí representaría problemas de sobreestimación en el modelo, por lo que debemos estar al pendiente de ello. De momento no haremos ningún tratamiento a estos outliers, simplemente debemos reconocerlos para saber que las medidas descriptivas (como la media, varianza o desviación estándar) están sesgadas y no son representativas, por lo que debemos emplear medidas robustas como la mediana o el MAD.</p>
          <p>Para reconocer si los outliers univariantes son también outliers en el espacio multivariante, debemos mirar las relaciones entre variables a través de gráficos de dispersión. Esto también nos dará noción de las relaciones subyacentes entre los datos, para poder elegir el mejor modelo. Así que a continuación se muestran gráficos de dispersión por pares, entre la que será nuestra variable dependiente "precio" y las variables independientes.</p>
          <p><strong>Diagramas de dispersión</strong></p>
             """,unsafe_allow_html=True)

#Diagramas de dispersion

Bases_Plot = [Base_T1,Base_T2,Base_T3]

for base_actual in Bases_Plot:
    n_vars = len(base_actual.columns) - 1  
    fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 4), sharey=True)
    if n_vars == 1:
        axes = [axes]  
    for j, var in enumerate(base_actual.columns[1:]):
        axes[j].scatter(base_actual[var], base_actual['precio'],color="#49007e")
        axes[j].set_title(f'Precio vs. {var}')
        axes[j].set_xlabel(var)
        axes[j].set_ylabel('Precio')
    
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("""
             <p>Mirando los gráficos de dispersión, podemos notar que sí existen outliers en el espacio multivariante, que son aquellos cuya superficie es mayor a los 200 metros cuadrados y también, aquellos cuyo precio es mayor a 20000 MXN, pues estos no siguen la estructura del resto de los datos. Se ha reconocido especialmente un outlier cuyo precio es mayor a 25000, lo he clasificado así, ya que en las variables observadas no presenta coherencia alguna que justifique un precio tan elevado. Esto puede deberse a dos cosas: la primera, es que posea algún atributo que no haya sido recogido en el muestreo como una variable y, por lo tanto, no nos permite explicar su precio; y por otro lado, podría ser simplemente un alquiler sobrevalorado que no responde a la lógica en el mercado de precios de alquileres. Con los diagramas de dispersión, también podemos darnos cuenta de que no existen relaciones claras entre las variables, no se comportan de manera lineal y tampoco de una forma fácilmente interpretable, así que tenemos relaciones complejas y no lineales, por lo que para hacer regresión necesitamos modelos capaces de abordar dicha complejidad en los datos. Sin embargo, antes de modelar vamos a revisar un modelo en la literatura para estos mismos datos, que fue planteado por los autores del libro de donde el conjunto de datos pertenece.</p>
            """,unsafe_allow_html=True)


st.markdown("""
         <p><h4>Literatura del caso</h4></p>
         <p>En su libro "<strong>Introducción a la valorización económica ambiental: teoría y práctica</strong>", Karina Caballero y Saúl Basurto estiman la disposición a pagar (DAP) por la belleza escénica en Tulum, medida a través del indicador vista al mar. Para ello, estiman un modelo de regresión lineal múltiple con forma funcional log-linear, donde el logaritmo natural del precio es la variable dependiente y vista al mar, superficie, capacidad, aire acondicionado, wifi, balcón, terraza, pueblo, TV, alberca, estacionamiento, gimnasio y bar son las variables independientes. Los resultados de dicha estimación son los siguientes:</p>
            """,unsafe_allow_html=True)


estimacion = pd.read_csv("Estimacion_mrl.csv")
st.write(estimacion)
st.markdown("<p>R2 = 0.6619, R2 ajustada = 0.6575, Varianza del modelo = 0.2263, Error Estandar = 0.4757, Estadistico F = 151.32 en 13 y 1005 grados de libertad -> p-value = 0.0<p>",unsafe_allow_html=True)

st.markdown("""<p>Caballero y Basurto encuentran que la DAP y el precio implícito de la vista al mar es del 19.43% del precio medio de los alquileres ($3,455), equivalente a $671.31 pesos. Al analizar las estimaciones, debemos notar que, en la regresión, las variables balcón y gimnasio resultan no significativas; el R<sup>2</sup> es de 0.66 (que además está en términos de la forma funcional log-lineal, ya que al transformar las estimaciones a la escala original de los datos y obtener el estadístico, este reduce su valor a 0.5485); y al replicar el modelo, este presenta problemas de normalidad, heterocedasticidad y autocorrelación, por lo cual, en estricto sentido, no es un modelo apto para la inferencia estadística. Lo cual se explica desde el análisis exploratorio, pues no se observó ningún tipo de relación lineal entre los datos. Cabe señalar que hacer una estimación del precio del alquiler no era el objetivo de Caballero y Basurto, sin embargo, es este el modelo al que vamos a intentar superar con un enfoque de Machine Learning.</p>
<p>Para ello, emplearemos los siguientes modelos:</p>
<ul>
  <li>SVR - Support Vector Regression (Regresiones de Soporte Vectorial)</li>
  <li>RFR - Random Forest Regression (Bosques Aleatorios de Regresión)</li>
  <li>XGBoost - eXtreme Gradient Boosting</li>
</ul>
<p>Estos modelos son capaces de capturar y modelar relaciones no lineales entre los datos, como los que se presentan en este conjunto, como ya lo revisamos anteriormente.</p>
            """,unsafe_allow_html=True)

st.markdown("""
            <h4>SVR - Support Vector Regression (Regresiones de Soporte Vectorial)</h4>
            <p>Son similares a las máquinas de soporte vectorial (SVM), solo que estas son empleadas para regresión. Estos modelos buscan estimar una variable continua en lugar de una etiqueta de clasificación (como lo haría un SVM). A diferencia de otros métodos de regresión que minimizan el error total, los SVR intentan ajustarse dentro de un margen de error establecido, y se enfocan en los puntos que están más cerca de la frontera de decisión, lo que les permite ser más eficientes en conjuntos de datos reducidos y logran adaptarse tanto a relaciones lineales como no lineales entre los datos. Sin embargo, no son los modelos para regresión más empleados dentro del Machine Learning.</p>
            <h4>RFR - Random Forest Regression (Bosques Aleatorios de Regresión)</h4>
            <p>Son modelos que bifurcan los datos para llegar a predicciones de una variable continua. Lo logran a través de preguntas sucesivas, basadas en las características de los datos, estos árboles guían la decisión hacia la hoja que ofrece la predicción final. Este enfoque los vuelve excelentes modelos para manejar complejidades y relaciones no lineales en los datos, adaptándose a variables numéricas y categóricas.</p>
            <h4>XGBoost (eXtreme Gradient Boosting)</h4>
            <p>Es resultado de una optimización avanzada del algoritmo de boosting del gradiente, el cual es un modelo predictivo conformado por muchos árboles de decisión. Dicho modelo funciona construyendo modelos secuenciales, donde cada nuevo modelo corrige errores cometidos por los modelos anteriores, ajustándose al gradiente del error respecto a la estimación. Los XGBoost se caracterizan por su velocidad y eficiencia en tiempo de ejecución y en el uso de memoria.</p>
            <h4>Estimaciones</h4>
            <p>Para realizar inferencia y probar el desempeño de cada uno de los modelos, he dejado una pequeña muestra de los datos de prueba. Estas observaciones corresponden a los datos reales. Elige algunos e introduce los valores de las variables independientes en el formulario que se encuentra debajo, con los datos de la siguiente tabla. Podrás realizar la estimación del <strong>precio</strong> con cada uno de los modelos mencionados (incluyendo regresión lineal) y comparar los resultados entre el valor estimado y el valor real para cada observación. Nota que mientras más bajas, mejor se vuelve el modelo ;).</p>
""",unsafe_allow_html=True)

pruebas = pd.read_csv("df_pruebas.csv")
pruebas.drop("Unnamed: 0", axis=1, inplace=True)
st.write(pruebas)
# Recolección de entradas del usuario
# Comienzo del formulario

with st.form(key='mi_formulario'):
    st.write('<p><h4>Introduce los atributos del alquiler (Haz click en la casilla si el valor en la variable es 1)</h4><p>', unsafe_allow_html=True)
    
    # Variables continuas y discretas dentro del formulario
    modelo_seleccionado = st.selectbox("Elige un modelo",["Regresión Lineal","SVR","RFR","XGBoost"])
    vistaalmar = st.checkbox('¿Tiene vista al mar?')
    superficie = st.number_input('Superficie (en metros cuadrados)', min_value=0.0, value=30.00, max_value=1000.0, step=1.00)
    capacidad = st.number_input('Capacidad (número de personas)', min_value=1, value=2, max_value=4)
    aire_ac = st.checkbox('¿Tiene aire acondicionado?')
    wifi = st.checkbox('¿Tiene Wi-Fi?')
    balcon = st.checkbox('¿Tiene balcón?')
    terraza = st.checkbox('¿Tiene terraza?')
    pueblo = st.checkbox('¿Está en el pueblo?')
    tv = st.checkbox('¿Tiene TV?')
    alberca = st.checkbox('¿Tiene alberca?')
    estacionamiento = st.checkbox('¿Tiene estacionamiento?')
    gym = st.checkbox('¿Tiene gimnasio?')
    bar = st.checkbox('¿Tiene bar?')

    # Botón de envío del formulario
    enviar = st.form_submit_button(label='Estimar con Modelos de Machine Learning')


# Procesar la entrada una vez que se envía el formulario
if enviar:
    precios = pd.read_csv("precios_totales.csv")

    #Cargar scaler y estandarizar datos de entrada
    pred = pd.Series([vistaalmar, superficie, capacidad, aire_ac, wifi, balcon, terraza, pueblo, tv, alberca, estacionamiento, gym,  bar])
    pred_df = pred.values.reshape(1, -1)
    pred_scale = scaler.transform(pred_df)

    #Modelo Lineal

    if modelo_seleccionado == "Regresión Lineal":
        prediccion_OLS = OLS.predict(pred_scale)
        prediccion_OLS = np.exp(prediccion_OLS)
        prediccion_OLS= prediccion_OLS[0]
        st.markdown("""
     <p><h4> Modelo - Regresión Lineal (OLS)</h4><p>
     <p><strong>Evaluación:</strong><p>
     <p><strong>MSE:</strong> 4009034.65, <strong>MEA:</strong> 1243.89, <strong>RMSE:</strong> 2002.25, <strong>:R²</strong>: 0.5485 <p>
     <p><strong>Estimación:</strong><p>
                """,unsafe_allow_html=True)
        st.write(f"El modelo de regresion lineal estima el precio de alquiler respectivo a los atributos proporcionados en <strong>{round(prediccion_OLS,2)} MXN </strong>",unsafe_allow_html=True)
        st.markdown("<p><strong>Visualización del rendimento del modelo:</strong><p>",unsafe_allow_html=True)
        precio_original = precios['precio_original']
        precio_estimado_lineal = precios['precio_estimado_lineal']
        observaciones = np.arange(len(precio_original))
        plt.figure(figsize=(12, 6))
        plt.scatter(observaciones, precio_original, color='#49007e', alpha=0.5, label='Precio Original')
        plt.scatter(observaciones, precio_estimado_lineal, color='#0a0310', alpha=0.5, label='Precio Estimado OLS')
        plt.title('Precios Originales vs. Precios Estimados por OLS')
        plt.xlabel('Observación')
        plt.ylabel('Precio')
        plt.legend()
        st.pyplot(plt)

        precios = pd.read_csv('precios_totales.csv')
        data = precios['error_lineal']
        density = gaussian_kde(data)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(data, bins=30, alpha=0.5, color='#49007e')
        ax[0].set_title('Hist. Residuos del modelo de regresion lineal')
        x = np.linspace(min(data), max(data), 1000)
        ax[1].plot(x, density(x), color='#0a0310')
        ax[1].set_title('Dens. Residuos del modelo de regresión lineal')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
                  <p>Al comparar las estimaciones del modelo de regresión lineal con los valores reales, habrás notado que la diferencia es abismal, y que para este conjunto de datos, este modelo no funciona debido a la no linealidad en las relaciones entre los datos. Además, al visualizar la distribución de los errores, notarás que está cargada a la derecha, con mayor peso en los errores positivos, lo que indica que el modelo subestimó, y que, en general, las estimaciones están por debajo del precio real. Las metricas de evaluación del error del modelo son muy altas, y su R<sup>2</sup> es muy bajo, por lo que no se desempeña correctamente.</p>
                """,unsafe_allow_html=True)
    

    #Modelo SVR

    elif modelo_seleccionado == "SVR":
        prediccion_SVR = SVR.predict(pred_scale)
        prediccion_SVR = np.exp(prediccion_SVR)
        prediccion_SVR= prediccion_SVR[0]
        st.markdown("""
     <p><h4> Modelo: SVR - Support Vector Regression (Regresiones de Soporte Vectorial)</h4><p>
     <p><strong>Evaluación:</strong><p>
     <p><strong>MSE:</strong> 3314195.50, <strong>MEA:</strong> 959.28, <strong>RMSE:</strong> 1820.49, <strong>:R²</strong>: 0.6268 <p>
     <p><strong>Estimación:</strong><p>
                """,unsafe_allow_html=True)
        st.write(f"El modelo SVR estima el precio de alquiler respectivo a los atributos proporcionados en <strong>{round(prediccion_SVR,2)} MXN </strong>",unsafe_allow_html=True)
        st.markdown("<p><strong>Visualización del rendimento del modelo:</strong><p>",unsafe_allow_html=True)
        precio_original = precios['precio_original']
        precio_estimado_SVR = precios['precio_estimado_SVR']
        observaciones = np.arange(len(precio_original))
        plt.figure(figsize=(12, 6))  # Ajusta el tamaño de la gráfica según necesites
        plt.scatter(observaciones, precio_original, color='#49007e', alpha=0.5, label='Precio Original')
        plt.scatter(observaciones, precio_estimado_SVR, color='#ffb238', alpha=0.5, label='Precio Estimado SVR')
        plt.title('Precios Originales vs. Precios Estimados por SVR')
        plt.xlabel('Observación')
        plt.ylabel('Precio')
        plt.legend()
        st.pyplot(plt)

        precios = pd.read_csv('precios_totales.csv')
        data = precios['error_SVG']
        density = gaussian_kde(data)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(data, bins=30, alpha=0.5, color='#49007e')
        ax[0].set_title('Hist. Residuos del modelo SVR')
        x = np.linspace(min(data), max(data), 1000)
        ax[1].plot(x, density(x), color='#ffb238')
        ax[1].set_title('Dens. Residuos del modelo SVR')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
                 <p>El modelo SVR ha mejorado significativamente los resultados del modelo de regresión lineal. Este no es precisamente el mejor modelo, pues aunque las métricas de error han disminuido significativamente, el R<sup>2</sup> de 0.62 sigue siendo poco aceptable. Si miras la gráfica de comparación de los precios y la distribución de los errores, notarás que este modelo tiene algunos errores positivos, que nos indican que no ha logrado ajustarse a los datos idóneamente, y sigue teniendo algunos problemas de subestimación. Este no es el modelo que mejores estimaciones realiza, pues aún tiene margen de mejora considerable.</p>
                """,unsafe_allow_html=True)
        
    elif modelo_seleccionado == "RFR":
        prediccion_RFR = RFR.predict(pred_scale)
        prediccion_RFR = np.exp(prediccion_RFR)
        prediccion_RFR= prediccion_RFR[0]
        st.markdown("""
     <p><h4> Modelo: RFR - Random Forest Regression (Bosques aleatorios de Regresión)</h4><p>
     <p><strong>Evaluación:</strong><p>
     <p><strong>MSE:</strong> 1536365.26, <strong>MEA:</strong> 815.84, <strong>RMSE:</strong> 1239.50, <strong>:R²</strong>: 0.8269 <p>
     <p><strong>Estimación:</strong><p>
                """,unsafe_allow_html=True)
        st.write(f"El modelo RFR estima el precio de alquiler respectivo a los atributos proporcionados en <strong>{round(prediccion_RFR,2)} MXN </strong>",unsafe_allow_html=True)
        st.markdown("<p><strong>Visualización del rendimento del modelo:</strong><p>",unsafe_allow_html=True)
        precio_original = precios['precio_original']
        precio_estimado_RFR = precios['precio_estimado_RFR']
        observaciones = np.arange(len(precio_original))
        plt.figure(figsize=(12, 6))  # Ajusta el tamaño de la gráfica según necesites
        plt.scatter(observaciones, precio_original, color='#49007e', alpha=0.5, label='Precio Original')
        plt.scatter(observaciones, precio_estimado_RFR, color='#ff005b', alpha=0.5, label='Precio Estimado RFR')
        plt.title('Precios Originales vs. Precios Estimados por RFR')
        plt.xlabel('Observación')
        plt.ylabel('Precio')
        plt.legend()
        st.pyplot(plt)

        precios = pd.read_csv('precios_totales.csv')
        data = precios['error_RFR']
        density = gaussian_kde(data)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(data, bins=30, alpha=0.5, color='#49007e')
        ax[0].set_title('Hist. Residuos del modelo RFR')
        x = np.linspace(min(data), max(data), 1000)
        ax[1].plot(x, density(x), color='#ff005b')
        ax[1].set_title('Dens. Residuos del modelo RFR')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
                 <p>El modelo RFR ha mejorado sus métricas en gran medida en comparación con los modelos de regresión lineal y SVR; el MSE, MAE y RMSE son mucho más bajos, y el R<sup>2</sup> ha mejorado a 0.82, lo cual es un gran indicador de evaluación. Al realizar estimaciones, los valores son bastante cercanos al valor real. Y si nos fijamos en la distribución de los errores, notamos cómo también tiene algunos problemas de sobrestimación, que sin embargo son bastante menores a los del modelo SVR.</p>
                """,unsafe_allow_html=True)

    #Modelo XGB

    elif modelo_seleccionado == "XGBoost":
    
        prediccion_XGB = XGB.predict(pred_scale)
        prediccion_XGB = np.exp(prediccion_XGB)
        prediccion_XGB= prediccion_XGB[0]
        prediccion_XGB_formateada = format(prediccion_XGB, ".2f")
        st.markdown("""
     <p><h4> Modelo: XGBoost (eXtreme Gradient Boosting)</h4><p>
     <p><strong>Evaluación:</strong><p>
     <p><strong>MSE:</strong> 857661.43, <strong>MEA:</strong> 563.08, <strong>RMSE:</strong> 926.10, <strong>:R²</strong>: 0.9034 <p>
     <p><strong>Estimación:</strong><p>
                """,unsafe_allow_html=True)
        st.write(f"El modelo XGB estima el precio de alquiler respectivo a los atributos proporcionados en <strong>{prediccion_XGB_formateada} MXN </strong> MXN",unsafe_allow_html=True)
        st.markdown("<p><strong>Visualización del rendimento del modelo:</strong><p>",unsafe_allow_html=True)
        precio_original = precios['precio_original']
        precio_estimado_XGB = precios['precio_estimado_XGB']
        observaciones = np.arange(len(precio_original))
        plt.figure(figsize=(12, 6))  # Ajusta el tamaño de la gráfica según necesites
        plt.scatter(observaciones, precio_original, color='#49007e', alpha=0.5, label='Precio Original')
        plt.scatter(observaciones, precio_estimado_XGB, color='#ff7d10', alpha=0.5, label='Precio Estimado XGB')
        plt.title('Precios Originales vs. Precios Estimados por XGB')
        plt.xlabel('Observación')
        plt.ylabel('Precio')
        plt.legend()
        st.pyplot(plt)

        precios = pd.read_csv('precios_totales.csv')
        data = precios['error_XGB']
        density = gaussian_kde(data)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(data, bins=30, alpha=0.5, color='#ff7d10')
        ax[0].set_title('Hist. Residuos del modelo de XGB')
        x = np.linspace(min(data), max(data), 1000)
        ax[1].plot(x, density(x), color='#49007e')
        ax[1].set_title('Dens. Residuos del modelo XGB')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
                <p>El modelo XGBoost ha hecho un gran trabajo en la estimación, como puede verse en las métricas, pues el MSE, MAE y RMSE son los más pequeños entre todos los modelos desarrollados, y el R<sup>2</sup> es de 0.9034, lo que es un excelente indicador. Las estimaciones son las más cercanas a los valores reales, y al mirar la distribución de los residuos, observamos que los problemas de subestimación se han reducido drasticamente, por lo que sin duda estamos frente a un gran modelo.</p>""",unsafe_allow_html=True)
        
        
    st.markdown("""
<h4>Conclusiones</h4>
                <p>Finalmente, hemos identificado la capacidad y superioridad de los modelos de regresión de Machine Learning frente a algunos modelos clásicos y tradicionales de la estadística. Ningún modelo es malo; simplemente, no todos los modelos son flexibles ni capaces de modelar relaciones complejas. Los modelos de Machine Learning se distinguen por su capacidad de abordar estas relaciones complejas y no lineales, por lo que se presentan como una gran alternativa para modelar fenómenos complejos de nuestra realidad. Aunque en general nuestros modelos presentaron problemas de subestimación, este se debe a que no dimos tratamiento a los outliers en el conjunto de datos, no realizamos dicho procedimento debido a que la intención de esta web es tambien mostrar el grado de robustes en los modelos de Machine Learning. A continuación se muestra una pequeña tabla comparativa entre los distintos modelos desarrollados y las metricas obtenidas por cada uno.</p>""",unsafe_allow_html=True)
    metricas_df = pd.read_csv("metricas.csv")
    st.write(metricas_df)
    st.markdown("""<h4>Referencias</h4>
                      <ul>
                         <li>Wu, D. C., Zhang, W., & Huang, X. (2010). Support Vector Regression for Predicting Customer Lifetime Value in E-commerce. Journal of the American Statistical Association, 105(472), 1177-1188.</li>
                         <li>Pal, S. K., & Mitra, S. (2009). Random Forest Ensemble Learning for Credit Scoring. Expert Systems with Applications, 36(6), 2669-2678.</li>
                         <li>Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Machine Learning Algorithm for Tree Boosting. In Proceedings of the 22nd ACM International Conference on Knowledge Discovery and Data Mining (pp. 785-796). ACM.</li>
                      </ul>""",unsafe_allow_html=True)
    
    


    
  



