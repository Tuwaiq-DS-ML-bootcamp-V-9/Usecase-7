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
st.image("https://i.postimg.cc/0jFymMHX/Screenshot-1446-08-24-at-2-36-53-PM.png", use_column_width=True)

# تحميل البيانات
file_path = "real_madrid_data.csv"
df = pd.read_csv(file_path)

# **👀 Overview of the Team**
st.markdown(f"<h2 style='color:{primary_color};'>📌 Team Overview</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Players", df.shape[0])

with col2:
    st.metric("Average Age", round(df["age"].mean(), 1))

with col3:
    st.metric("Average Height (cm)", round(df["height"].mean(), 1))

# **📊 Player Position Distribution**
st.markdown(f"<h3 style='color:{secondary_color};'>⚽ Player Position Distribution</h3>", unsafe_allow_html=True)
position_counts = df["position"].value_counts()
fig, ax = plt.subplots()
ax.bar(position_counts.index, position_counts.values, color=primary_color)
plt.xticks(rotation=45)
plt.xlabel("Position", fontsize=12)
plt.ylabel("Number of Players", fontsize=12)
plt.title("Distribution of Players by Position", fontsize=14)
st.pyplot(fig)

# **💰 Market Value of Players**
st.markdown(f"<h2 style='color:{primary_color};'>💰 Market Value of Players</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    most_valuable_player = df.loc[df["current_value"].idxmax()]
    st.metric("Most Valuable Player", most_valuable_player["name"], f"{most_valuable_player['current_value']} Million €")

with col2:
    highest_value_player = df.loc[df["highest_value"].idxmax()]
    st.metric("Highest Market Value Ever", highest_value_player["name"], f"{highest_value_player['highest_value']} Million €")

# **📉 Market Value Changes**
st.markdown(f"<h3 style='color:{secondary_color};'>📊 Market Value Changes</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["highest_value"], df["current_value"], color=secondary_color)
plt.xlabel("Highest Market Value (€ Million)", fontsize=12)
plt.ylabel("Current Market Value (€ Million)", fontsize=12)
plt.title("How Player Values Changed?", fontsize=14)
st.pyplot(fig)

# **🎯 Player Performance**
st.markdown(f"<h2 style='color:{primary_color};'>🎯 Player Performance</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    top_scorer = df.loc[df["goals"].idxmax()]
    st.metric("Top Scorer", top_scorer["name"], f"{top_scorer['goals']} Goals")

with col2:
    top_assist = df.loc[df["assists"].idxmax()]
    st.metric("Top Playmaker", top_assist["name"], f"{top_assist['assists']} Assists")

# **🏥 Impact of Injuries**
st.markdown(f"<h2 style='color:{alert_color};'>🏥 Impact of Injuries</h2>", unsafe_allow_html=True)
st.metric("Total Days Lost Due to Injury", df["days_injured"].sum())
st.metric("Total Games Missed Due to Injury", df["games_injured"].sum())

# **🏆 Awards & Achievements**
st.markdown(f"<h2 style='color:{primary_color};'>🏆 Awards & Achievements</h2>", unsafe_allow_html=True)
award_counts = df["award"].value_counts()
if not award_counts.empty:
    fig, ax = plt.subplots()
    ax.bar(award_counts.index, award_counts.values, color=primary_color)
    plt.xlabel("Number of Awards", fontsize=12)
    plt.ylabel("Number of Players", fontsize=12)
    plt.title("Awards Won by Players", fontsize=14)
    st.pyplot(fig)
else:
    st.write("⚠️ No award data available.")

# **💡 Explore Data Interactively**
st.markdown(f"<h2 style='color:{primary_color};'>📊 Explore Data Interactively!</h2>", unsafe_allow_html=True)
st.dataframe(df)
