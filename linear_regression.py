import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
st.title("Linear Regression")
st.header("What is Linear Regression?什麼是線性迴歸?")
st.text("還記得函數(function)的形式: y = mx + b或y= ax + b")
st.text("這都不重要")
st.text("重要的是本身是用一個已知的值推估未知的值")
st.text("神奇的來了!怎麼這樣可以預測呢?")
st.text("這就是機器學習的魔法啊!")
st.text("這關乎到數學裡的統計學")
st.text("我們把它繪製成以下的圖表就一目了然了")

def get_data():
    data = pd.read_excel("regression_data.xlsx")
    return data

def sidebar_func(data):
    weight = data["weight(kg)"]
    binding = data["Seated Forward Bend(cm)"]
    jumping = data["Standing long jump(cm)"]
    situps = data["sit-up(once)"]
    with st.sidebar:
        st.header("篩選資料")
        st.write("選擇要預測的資料")
        selection = st.selectbox("選擇資料", ["體重(公斤)", "坐姿體前彎(公分)", "立定跳遠(公分)", "仰臥起坐(下)"])
    if selection == "體重(公斤)":
        
        return weight
    elif selection == "坐姿體前彎(公分)":
       
        return binding
    elif selection == "立定跳遠(公分)":
        
        return jumping
    else:  
        
        return situps
data = get_data()
value = sidebar_func(data)





x= data["height(cm)"]
y = value
x = x.values.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_
fig, ax = plt.subplots()
x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)
ax.plot(x_range, y_pred, color='red', label=f'Regression line: y = {slope:.2f}x + {intercept:.2f}')
plt.scatter(x, y, color='blue', label='Data points')
ax.set_xlabel("height(cm)")
value = pd.DataFrame(value)
ax.set_ylabel(value.columns[0])
ax.set_title('Simple Linear Regression with scikit-learn')
ax.legend()

# 在 Streamlit 中显示图形
st.pyplot(fig, dpi=300)
