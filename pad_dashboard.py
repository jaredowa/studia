import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(layout = 'wide', page_title = 'Diamonds Dashboard', initial_sidebar_state='expanded')

st.markdown(
    """
<style>

.main {
  background-color: #facfe2;
}
</style>
""",
    unsafe_allow_html=True,
)

df = pd.read_csv('messy_data.csv')
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df = df.rename(columns=lambda x: x.strip())
df.columns = df.columns.str.replace(' ', '_')
df['clarity'] = df['clarity'].str.upper()
df['color'] = df['color'].str.upper()
df['color'] = df['color'].replace('COLORLESS', 'D')
df['cut'] = df['cut'].str.capitalize()
df = df.replace('', np.NaN)
df['depth'] = df['depth'].astype(float)
df['table'] = df['table'].astype(float)
df['x_dimension'] = df['x_dimension'].astype(float)
df['y_dimension'] = df['y_dimension'].astype(float)
df['z_dimension'] = df['z_dimension'].astype(float)
df['price'] = df['price'].astype(float)
df['x_dimension'] = df['x_dimension'].fillna(round((df['z_dimension'] + df['y_dimension']) / 2, 2))
df['y_dimension'] = df['y_dimension'].fillna(round((df['x_dimension'] + df['z_dimension']) / 2, 2))
df['z_dimension'] = df['z_dimension'].fillna(round((df['x_dimension'] + df['y_dimension']) / 2, 2))
df['depth'] = df['depth'].fillna(round((df['z_dimension'] / ((df['x_dimension'] + df['y_dimension']) / 2)) * 100, 2))
df['table'] = df['table'].fillna(round((df['x_dimension'] / df['y_dimension']) * 100, 2))
df['carat'] = df['carat'].fillna(round(df['x_dimension'] * np.mean(df['carat'] / df['x_dimension']), 2))

carat_Q1 = df['carat'].quantile(0.25)
carat_Q3 = df['carat'].quantile(0.75)
carat_IQR = carat_Q3 - carat_Q1
df = df[(df['carat'] >= carat_Q1 - 1.5 * carat_IQR) & (df['carat'] <= carat_Q3 + 1.5 * carat_IQR)]

price_Q1 = df['price'].quantile(0.25)
price_Q3 = df['price'].quantile(0.75)
price_IQR = price_Q3 - price_Q1
df = df[(df['price'] >= price_Q1 - 1.5 * price_IQR) & (df['price'] <= price_Q3 + 1.5 * price_IQR)]

depth_Q1 = df['depth'].quantile(0.25)
depth_Q3 = df['depth'].quantile(0.75)
depth_IQR = depth_Q3 - depth_Q1
df = df[(df['depth'] >= depth_Q1 - 1.5 * depth_IQR) & (df['depth'] <= depth_Q3 + 1.5 * depth_IQR)]

table_Q1 = df['table'].quantile(0.25)
table_Q3 = df['table'].quantile(0.75)
table_IQR = table_Q3 - table_Q1
df = df[(df['table'] >= table_Q1 - 1.5 * table_IQR) & (df['table'] <= table_Q3 + 1.5 * table_IQR)]

z_dimension_Q1 = df['z_dimension'].quantile(0.25)
z_dimension_Q3 = df['y_dimension'].quantile(0.75)
z_dimension_IQR = z_dimension_Q3 - z_dimension_Q1
df = df[(df['z_dimension'] >= z_dimension_Q1 - 1.5 * z_dimension_IQR) & (df['z_dimension'] <= z_dimension_Q3 + 1.5 * z_dimension_IQR)]

st.header('Cleaned Data Frame')

st.dataframe(df)

page = st.sidebar.selectbox('Select a plot', ['Variables vs Price', 'Variables Distribution', 'Carat & Clarity vs Price', 'Correlation Matrix'])

if page == 'Variables vs Price':
    sel_list = st.selectbox('Select a variable', options=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x_dimension', 'y_dimension', 'z_dimension'])
    fig = px.scatter(df, x=sel_list, y='price', title=f'{sel_list} vs price')
    fig.update_traces(marker=dict(size=9, color='pink'))
    fig.update_layout(title=f'{sel_list} vs price', xaxis_title=sel_list, yaxis_title='price', title_x=0.42, title_y=0.9, title_font_size=18, title_font_family='Arial', title_font_color='purple')
    st.plotly_chart(fig)
if page == 'Variables Distribution':
    sel_list = st.selectbox('Select a variable', options=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x_dimension', 'y_dimension', 'z_dimension'])
    fig = px.histogram(df, x=sel_list, title=f'{sel_list} distribution')
    fig.update_traces(marker=dict(color='pink'))
    fig.update_layout(title=f'{sel_list} distribution', xaxis_title=sel_list, yaxis_title='price', title_x=0.42, title_y=0.9, title_font_size=18, title_font_family='Arial', title_font_color='purple')
    st.plotly_chart(fig)
if page == 'Carat & Clarity vs Price':
    df['clarity'].replace(['IF', 'VVS1', 'VVS2', 'SI1', 'SI2', 'I1'],
                        [5, 4, 3, 2, 1, 0], inplace=True)
    df['color'].replace(['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                        [6, 5, 4, 3, 2, 1, 0], inplace=True)
    df['cut'].replace(['Fair', 'Good', 'Very good', 'Premium', 'Ideal'],
                        [0, 1, 2, 3, 4], inplace=True)
    fig = px.scatter(df, x='carat', y='price', color='clarity', marginal_y='violin', title='Carat & Clarity vs Price')
    fig.update_layout(title='Carat & Clarity vs Price', xaxis_title='carat', yaxis_title='price', title_x=0.35, title_y=0.9, title_font_size=18, title_font_family='Arial', title_font_color='purple')
    st.plotly_chart(fig)
if page == 'Correlation Matrix':
    df['clarity'].replace(['IF', 'VVS1', 'VVS2', 'SI1', 'SI2', 'I1'],
                        [5, 4, 3, 2, 1, 0], inplace=True)
    df['color'].replace(['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                        [6, 5, 4, 3, 2, 1, 0], inplace=True)
    df['cut'].replace(['Fair', 'Good', 'Very good', 'Premium', 'Ideal'],
                        [0, 1, 2, 3, 4], inplace=True)
    fig = px.imshow(df.corr(), color_continuous_scale='purples', title='Correlation Matrix')
    fig.update_layout(title='Correlation Matrix', title_x=0.39, title_y=0.9, title_font_size=18, title_font_family='Arial', title_font_color='purple')
    st.plotly_chart(fig)

