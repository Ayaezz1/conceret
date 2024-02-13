import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import sys


st.markdown("<h2 style='text-align: left; font-size: 14px; color: black'>Date: February 22,2024 </h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left; font-size: 14px; color: black'>Student Name: Aya Ezzedine </h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: Center; font-size: 30px; color: black'>Assignment 3- Building Interactive Visualizations with Streamlit </h2>", unsafe_allow_html=True)

# Load the data
df = pd.read_csv(r"C:\\Users\\ayaez\\Desktop\\Courses 2023\\Semester 2\\MSBA225_Data visualization and communication\\Assignment 2\\concrete.csv")

#Cleaning the Data from any Outliers We had 1030 data before cleaning. 

df = df.drop(df[df['slag'] > 350].index)
df = df.drop(df[(df['water'] < 135) | (df['water'] > 230)].index)
df = df.drop(df[df['superplastic'] > 25].index)
df = df.drop(df[(df['fineagg'] < 640) | (df['fineagg'] > 925)].index)
df = df.drop(df[df['age'] > 80].index)
df = df.drop(df[df['strength'] > 71].index)

#Displaying the Data After Cleaning 
st.markdown("<h2 style='text-align: left; font-size: 30px; color: red'> I. Visualizations 1 -Displaying Data Table with Interactive Features</h2>", unsafe_allow_html=True)

# Interactive feature 1: Slider to select number of rows to display
num_rows_to_display = st.slider("Select number of rows to display:", 1, len(df), value=5)

# Interactive feature 2: Checkbox to toggle data sorting
sort_by_column = st.checkbox("Sort data by selected column")
if sort_by_column:
    column_to_sort = st.selectbox("Select column to sort by:", df.columns)
    df = df.sort_values(by=column_to_sort)

# Display the data with interactivity 
st.markdown("<h2 style='text-align: left; font-size: 24px; color: blue'> Experimental Data </h2>", unsafe_allow_html=True)
st.dataframe(df.head(num_rows_to_display))  # Display selected number of rows

st.markdown("<h2 style='text-align: left; font-size: 20px; color: blue'>Press Below for Full Data</h2>", unsafe_allow_html=True)
st.download_button("Download CSV", df.to_csv())



st.markdown("<h2 style='text-align: left; font-size: 30px; color: red'> II. Visualizations 2 - Data Boxplot with Interactive Features </h2>", unsafe_allow_html=True)

# Interactive feature 1: Dropdown for selecting variables
selected_variable = st.selectbox("Select variable for boxplot:", df.columns)

# Interactive feature 2: Slider for adjusting boxplot width
boxplot_width = st.slider("Adjust boxplot width", 1, 40, value=20)

# Creating the boxplot with interactivity
fig, ax = plt.subplots(figsize=(boxplot_width, 5))
sns.boxplot(data=df, x=selected_variable, ax=ax)
plt.title("Boxplot of {}".format(selected_variable))

# Display the plot in Streamlit
st.pyplot(fig)


st.markdown("<h2 style='text-align: left; font-size: 30px; color: red'> III. Visualizations 3 - Scatter Chart with Interactive Features </h2>", unsafe_allow_html=True)

# Create the interactive scatter plot
fig_px = px.scatter(df, x='cement', y='strength')

# Define hover function (same as before)
def hover_func(trace, point_num, data):
    cement_value = df['cement'].iloc[point_num]
    strength_value = df['strength'].iloc[point_num]
    return f"Cement: {cement_value:.2f}<br>Strength: {strength_value:.2f}"

hover_template = hover_func(fig_px.data[0], 0, fig_px.data)
fig_px.update_traces(hovertemplate=hover_template)

# Interactive feature 1: Dropdown to select x-axis variable
x_axis_variable = st.selectbox("Select x-axis variable:", df.columns)
fig_px.update_layout(xaxis_title=x_axis_variable)

# Interactive feature 2: Checkbox to toggle regression line
show_regression = st.checkbox("Show regression line")
if show_regression:
    fig_px.add_traces(px.scatter(df, x=x_axis_variable, y='strength', trendline='ols'))

st.plotly_chart(fig_px, use_container_width=True)


st.markdown("<h2 style='text-align: left; font-size: 30px; color: red'> III. Visualizations 4 - Heat Map with Interactive Features </h2>", unsafe_allow_html=True)

# Interactive feature 1: Dropdown to select correlation method
correlation_method = st.selectbox("Select correlation method:", ['pearson', 'spearman', 'kendall'])

# Interactive feature 2: Slider to adjust mask threshold
mask_threshold = st.slider("Adjust mask threshold", 0.0, 1.0, value=0.5)

# Interactive feature 3: Checkbox to highlight selected cells
highlight_selected = st.checkbox("Highlight selected cells", value=False)

if highlight_selected:
    cell_to_highlight = st.multiselect("Select cells to highlight:", df.columns)

# Create the heatmap with interactivity
fig, ax = plt.subplots(figsize=(16, 8))
corr_matrix = df.corr(method=correlation_method)  

# Apply mask and highlight if needed
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
mask[np.abs(corr_matrix) < mask_threshold] = False

if highlight_selected:
    for row, col in zip(*np.argwhere(corr_matrix.index.isin(cell_to_highlight) & corr_matrix.columns.isin(cell_to_highlight))): mask[row, col] = True 

sns.heatmap(corr_matrix, annot=True, cmap='Purples', mask=mask, ax=ax)
plt.title('Correlation among features on concrete mixture', fontsize=20)

st.pyplot(fig)









