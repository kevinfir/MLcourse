import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
st.title("機器學習是一門科學，其目的是賦予計算機學習能力，且不需要明確的編寫程式")
st.header("機器學習系統進行學習做出決策的部分稱為模型")
st.text("模型是一個數學函數，它將輸入映射到輸出")
st.text("線性迴歸是一種簡單的機器學習模型")
st.text("使用現有的數據來預測未來的數據或找出數據之間的規則")
st.text("機器學習的應用範例")
st.text("1.分析產線的瑕疵品，常用CNN(convolutional neural network)卷積神經網絡，有時用Transformer")
st.text("2.斷層掃描圖像的分析，常用CNN(convolutional neural network)卷積神經網絡")
st.text("3.新聞文章的分類，通常都用NLP的框架，常用RNN(recurrent neural network)循環神經網絡")
st.text("4.信用卡詐騙偵測，常用GVM(gaussian mixture model)高斯混合模型，隨機樹(Isolation Forest)")
st.text("6.遊戲中的機器人，常用Q-learning，DQN(deep Q-network)深度Q網絡，或RL(reinforcement learning)強化學習")
st.header("機器學習的種類")
st.text("1.監督式學習(supervised learning)")
st.text("手動上標籤的數據集，標籤是人工標記的")
st.text("2.非監督式學習(unsupervised learning)")
st.text("沒有標籤的數據集，模型自己學習")
st.text("3.半監督式學習(semi-supervised learning)")
st.text("部分數據有標籤，部分數據沒有標籤，或是先非監督式學習學到一個程度再進行監督式學習")
st.text("4.自我監督學習(self-supervised learning)")
st.text("透過無標籤的資料進行學習，透過獎勵機制來學習，最後讓機器自己上標籤")
st.text("5.強化學習(reinforcement learning)")
st.text("透過獎勵機制來學習，通常用於遊戲中的機器人")
st.text("主要是透過代理人來遊玩遊戲，透過獎勵機制來學習")
st.text("代理人主要是一個AI模型")
st.header("批次學習與線上學習")
st.text("批次學習(batch learning)")
st.text("又稱離線學習，是指在訓練模型時，將所有數據一次性輸入模型，訓練完暫時不再更新")
st.text("線上學習(online learning)")
st.text("是指在訓練模型時，將數據分批次輸入模型，訓練完後會不斷更新訓練資料")
st.text("線上學習適合大數據，批次學習適合小數據")
st.text("線上學習的優點是可以不斷更新模型，但缺點是需要大量的計算資源")
url = "https://www.youtube.com/watch?v=HsLup7yy-6I"
st.link_button("但業界曾經推出線上學習如微軟的tay",url)

st.title("Linear Regression")
st.header("What is Linear Regression?什麼是線性迴歸?")
st.text("還記得函數(function)的形式")
st.text("y = mx + b或y= ax + b")
st.text("這都不重要")
st.text("重要的是本身是用一個已知的值推估未知的值")
st.text("神奇的來了!怎麼這樣可以預測呢?")
st.text("這就是機器學習的魔法啊!")
st.text("這關乎到數學裡的統計學")
st.text("線性迴歸就是一種統計學的方法")
st.text("假設某一個值增加另外一個值也會增加")
st.text("分析兩者的關係，假設關係成立")
st.text("我們就會認為它呈現互相增長的關係")
st.text("而我們就認為這些數據是可以預測的喔")
st.text("我們把它繪製成以下的圖表就一目了然了")
st.markdown(":blue[test]")
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
    

lifesat = pd.read_csv("lifesat.csv")
x = lifesat[["GDP per capita (USD)"]].values.reshape(-1, 1)
y = lifesat[["Life satisfaction"]].values
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(x, y)

# Predict values
x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred = knn.predict(x_range)

# Plot results
fig, ax = plt.subplots()
ax.scatter(x, y, color='blue', label='Data points')
ax.plot(x_range, y_pred, color='red', label='KNN regression')
ax.legend()
ax.set_xlabel("GDP per capita (USD)")
ax.set_ylabel("Life satisfaction")
ax.set_title("KNN Regression: Life Satisfaction vs. GDP per Capita")

# Display plot in Streamlit
st.pyplot(fig)
fig, ax = plt.subplots()
lifesat.plot(kind='scatter', x="GDP per capita (USD)", y='Life satisfaction', title='Life Satisfaction vs. GDP per Capita', ax=ax)
ax.axis([23_500, 62_500, 4, 9])


model = LinearRegression()
model.fit(x, y)
slope = model.coef_[0][0]
intercept = model.intercept_[0]
x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)
ax.plot(x_range, y_pred, color='red', label=f'Regression line: y = {slope:.2f}x + {intercept:.2f}')
plt.scatter(x, y, color='blue', label='Data points')
y_pred_original = model.predict(x)
for i in range(len(x)):
    ax.plot([x[i], x[i]], [y[i], y_pred_original[i]], color='green', linestyle='--', linewidth=0.8)
ax.legend()
st.pyplot(fig, dpi=300)
text = st.text_input("輸入GDP per capita (USD)的值")

if text:
    text = float(text)
    result = model.predict([[text]])[0][0]
    st.write(f"Life satisfaction: {result:.2f}")

data = get_data()
st.table(data)
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
y_pred_original = model.predict(x)
for i in range(len(x)):
    ax.plot([x[i], x[i]], [y[i], y_pred_original[i]], color='green', linestyle='--', linewidth=0.8)
ax.set_xlabel("height(cm)")
value = pd.DataFrame(value)
ax.set_ylabel(value.columns[0])
ax.set_title('Simple Linear Regression with scikit-learn')
ax.legend()

# 在 Streamlit 中显示图形
st.pyplot(fig, dpi=300)


