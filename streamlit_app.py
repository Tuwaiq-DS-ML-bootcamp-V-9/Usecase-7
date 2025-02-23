import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# إعداد ألوان ريال مدريد
primary_color = "#004996"  # أزرق ملكي
secondary_color = "#FCBF00"  # ذهبي
background_color = "#FFFFFF"  # أبيض
alert_color = "#E62644"  # أحمر داكن

# إعداد الصفحة
st.set_page_config(page_title="تحليل ريال مدريد", page_icon="⚽", layout="wide")

# إضافة بانر ريال مدريد
st.image("https://k.top4top.io/p_334107ni1.png", use_column_width=True)

# تحميل البيانات
file_path = "real_madrid_data.csv"  # اسم الملف المرفوع
df = pd.read_csv(file_path)

# **👀 أولاً: نظرة على الفريق**
st.markdown(f"<h2 style='color:{primary_color};'>📌 نظرة على الفريق</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("عدد اللاعبين", df.shape[0])

with col2:
    st.metric("متوسط العمر", round(df["age"].mean(), 1))

with col3:
    st.metric("متوسط الطول (سم)", round(df["height"].mean(), 1))

# **📊 توزيع اللاعبين حسب المراكز**
st.markdown(f"<h3 style='color:{secondary_color};'>⚽ توزيع اللاعبين حسب المراكز</h3>", unsafe_allow_html=True)
position_counts = df["position"].value_counts()
fig, ax = plt.subplots()
ax.bar(position_counts.index, position_counts.values, color=primary_color)
plt.xticks(rotation=45)
plt.xlabel("المركز")
plt.ylabel("عدد اللاعبين")
st.pyplot(fig)

# **💰 قيمة لاعبي ريال مدريد في سوق الانتقالات**
st.markdown(f"<h2 style='color:{primary_color};'>💰 قيمة اللاعبين في سوق الانتقالات</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    most_valuable_player = df.loc[df["current_value"].idxmax()]
    st.metric("أغلى لاعب حاليًا", most_valuable_player["name"], f"{most_valuable_player['current_value']} مليون €")

with col2:
    highest_value_player = df.loc[df["highest_value"].idxmax()]
    st.metric("أعلى قيمة تاريخية", highest_value_player["name"], f"{highest_value_player['highest_value']} مليون €")

# **📉 مقارنة بين أعلى قيمة سوقية والقيمة الحالية**
st.markdown(f"<h3 style='color:{secondary_color};'>📊 كيف تغيرت قيمة اللاعبين؟</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["highest_value"], df["current_value"], color=secondary_color)
plt.xlabel("أعلى قيمة سوقية (مليون €)")
plt.ylabel("القيمة الحالية (مليون €)")
plt.title("تغير القيم السوقية للاعبين")
st.pyplot(fig)

# **🎯 أداء اللاعبين في المباريات**
st.markdown(f"<h2 style='color:{primary_color};'>🎯 أداء اللاعبين في المباريات</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    top_scorer = df.loc[df["goals"].idxmax()]
    st.metric("الهداف", top_scorer["name"], f"{top_scorer['goals']} هدف")

with col2:
    top_assist = df.loc[df["assists"].idxmax()]
    st.metric("أفضل صانع ألعاب", top_assist["name"], f"{top_assist['assists']} أسيست")

# **🏥 تأثير الإصابات على الفريق**
st.markdown(f"<h2 style='color:{alert_color};'>🏥 تأثير الإصابات على اللاعبين</h2>", unsafe_allow_html=True)
st.metric("إجمالي الأيام الضائعة بسبب الإصابات", df["days_injured"].sum())
st.metric("إجمالي المباريات التي فاتتها اللاعبين بسبب الإصابات", df["games_injured"].sum())

# **🏆 الجوائز والإنجازات**
st.markdown(f"<h2 style='color:{primary_color};'>🏆 الجوائز والإنجازات</h2>", unsafe_allow_html=True)
award_counts = df["award"].value_counts()
if not award_counts.empty:
    st.bar_chart(award_counts)
else:
    st.write("⚠️ لا توجد بيانات كافية عن الجوائز.")

# **📌 التوصيات النهائية**
st.markdown(f"<h2 style='color:{secondary_color};'>📌 التوصيات والتحليلات النهائية</h2>", unsafe_allow_html=True)
st.write("""
🔹 هل يحتاج ريال مدريد إلى تعزيزات جديدة في سوق الانتقالات؟  
🔹 من هم اللاعبين الأكثر تأثيرًا في الفريق؟  
🔹 كيف يمكن للنادي تقليل الإصابات وتحسين أداء اللاعبين؟
""")

# **💡 التفاعل مع البيانات**
st.markdown(f"<h2 style='color:{primary_color};'>💡 استكشف البيانات بنفسك!</h2>", unsafe_allow_html=True)
st.dataframe(df)


