import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="تحليل أسعار انتقالات اللاعبين", layout="wide")

# إضافة كود الـCSS لتطبيق الاتجاه والترتيب على مستوى التطبيق والنماذج
st.markdown(
    """
    <style>
    /* نجعل كامل التطبيق يعرض من اليمين لليسار ويكون مبرر */
    [data-testid="stAppViewContainer"] {
        direction: rtl;
        text-align: justify;
        unicode-bidi: bidi-override;
    }

    /* استهداف حاوية الMarkdown الافتراضية لمحاذاة النص */
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
        text-align: justify;
        direction: rtl;
        unicode-bidi: bidi-override;
    }

    /* قلب اتجاه النموذج، مع إبقاء النص عاليمين */
    [data-testid="stForm"] {
        text-align: right; 
    }

    /* جعل كل اسم حقل (label) في سطر لوحده */
    [data-testid="stForm"] label {
        float: none !important;
        display: block !important;
        text-align: right !important;
        margin-bottom: 0.3rem;
    }

    /* جعل حقول الإدخال (أرقام، نص، خيارات...) في سطر مستقل */
    [data-testid="stForm"] input,
    [data-testid="stForm"] select,
    [data-testid="stForm"] textarea {
        float: none !important;
        display: block !important;
        text-align: right !important;
        margin-bottom: 1rem; /* مسافة تحت الحقل */
    }

    /* زر التوقع أو أي أزرار أخرى تظهر على اليمين */
    [data-testid="stForm"] button {
        float: right;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Data Loading ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data/cleaned_football.csv")
    df['has_award'] = df['award'] > 0  # Convert awards to boolean
    df['value_retention'] = (df['current_value'] / df['highest_value']) * 100  # Calculate value retention
    return df

df = load_data()

# ---------------- Model and Feature Scaler Loading ----------------
@st.cache_resource
def load_models():
    # Load trained models.
    gk_model = joblib.load("models/goalkeeper_model.pkl")
    def_model = joblib.load("models/defender_model.pkl")
    mid_model = joblib.load("models/midfielder_model.pkl")
    fw_model = joblib.load("models/attacker_model.pkl")
    
    # Load the feature scalers (used during training).
    gk_scaler = joblib.load("scalars/goalkeeper_scaler.pkl")
    def_scaler = joblib.load("scalars/defender_scaler.pkl")
    mid_scaler = joblib.load("scalars/midfielder_scaler.pkl")
    fw_scaler = joblib.load("scalars/attacker_scaler.pkl")
    
    return (gk_model, def_model, mid_model, fw_model,
            gk_scaler, def_scaler, mid_scaler, fw_scaler)

(gk_model, def_model, mid_model, fw_model,
 gk_scaler, def_scaler, mid_scaler, fw_scaler) = load_models()

# ---------------- Sidebar Navigation ----------------
analysis_option = st.sidebar.radio(
    "اختر ما تريد:",
    [
        "🏠 الصفحة الرئيسية",
        "📌 كيف تختلف أسعار الانتقالات بين المراكز؟",
        "📌 العمر.. هل هو مجرد رقم؟",
        "📌 الجوائز.. هل فعلاً ترفع سعر اللاعب؟",
        "📌 الإصابات.. مين أكثر مركز يعاني؟",
        "📌 هل اللاعب يحافظ على قيمته بمرور الوقت؟",
        "🔮 توقع سعر اللاعب"
    ]
)

# ---------------- Page Sections ----------------
if analysis_option == "🏠 الصفحة الرئيسية":
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        <h1>💰 وش السالفة؟ كيف تتحدد أسعار اللاعبين؟ ⚽🔥</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        🤯 هل قد سألت نفسك ليه بعض اللاعبين يُدفع فيهم ملايين، بينما غيرهم بالكاد يحصلون على عقد كويس؟<br>
        اليوم، بنفك لك الموضوع بالكامل! 🎯 بنشوف كيف الأرقام والإحصائيات داخل الملعب هي الأساس في تحديد قيمة اللاعب، 
        وبنحط النقاط على الحروف علشان تفهم ليه بعض النجوم أسعارهم تطير فوق! 😏💸
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        الأهداف، التمريرات الحاسمة، نسب الاستحواذ، وحتى تفاصيل صغيرة زي معدل الركض وقطع الكرات—كلها تتحول إلى معادلات رقمية 
        ترفع لاعب عادي، أو تهمّش نجم ما وثّق أرقامه. في هالتقرير، نكشف لك كيف هالبيانات تصنع ملايين اللاعبين، وتخلينا نشوف 
        السعر “الحقيقي” لكل موهبة على أرض الملعب.
    </div>
    """, unsafe_allow_html=True)

elif analysis_option == "📌 كيف تختلف أسعار الانتقالات بين المراكز؟":
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        <h2>📌 كيف تختلف أسعار الانتقالات بين المراكز؟ 🤔</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        تخيل إنك تفتح مزاد، هل بتدفع نفس المبلغ لحارس مرمى، ومهاجم؟ أكيد لا! 😅
    </div>
    """, unsafe_allow_html=True)
    
    mean_value_by_position = df.groupby('position_label')['current_value'].mean().sort_values(ascending=False)
    fig = px.bar(
        x=mean_value_by_position.index,
        y=mean_value_by_position.values,
        labels={'x': "المركز", 'y': "متوسط قيمة الانتقال (€M)"},
        color=mean_value_by_position.values,
        color_continuous_scale="teal"
    )
    st.plotly_chart(fig)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        💡 <b>واضح من الرسم إن المهاجمين ياخذون أعلى قيمة في الانتقالات مقارنةً بباقي المراكز، بينما حراس المرمى أقلهم. 
        السبب بسيط: الأندية عادةً تدفع أكثر للِّي يسجّل الأهداف، لأنهم يعتبرون العنصر الهجومي مفتاح الفوز وحصد النقاط.</b>
    </div>
    """, unsafe_allow_html=True)

elif analysis_option == "📌 العمر.. هل هو مجرد رقم؟":
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        <h2>📌 العمر.. هل هو مجرد رقم؟ ⏳</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        كثير ناس تقول "العمر مجرد رقم"، لكن في سوق الانتقالات؟ العمر حرفيًا يحدد مصير اللاعب! 😲
    </div>
    """, unsafe_allow_html=True)
    
    fig = px.scatter(
        df, x="age", y="current_value",
        labels={"age": "العمر", "current_value": "القيمة السوقية (€M)"},
        color="current_value",
        color_continuous_scale="teal"
    )
    st.plotly_chart(fig)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        💡 <b>من الرسم يظهر إن أغلب الأسعار الأعلى تتركّز في فئة اللاعبين اللي أعمارهم بين بداية العشرينات 
        حتى أواخر العشرينات؛ هذا هو السن اللي غالبًا يكون اللاعب فيه في قمّة مستواه وبكامل جاهزيته بدنياً ومهارياً. 
        بعد سن الـ30 تقريبًا، نشوف تراجع ملحوظ في القيمة السوقية، يعكس نظرة الأندية لحجم العطاء المتبقي واحتمالية الإصابات 
        وتقلّص السرعة والجهد على المدى الطويل. بمعنى آخر، كلّ ما اقترب اللاعب من الثلاثين وما بعدها، قلّت الأرقام 
        الفلكية وبدأ خطّ السعر بالهبوط التدريجي.</b>
    </div>
    """, unsafe_allow_html=True)

elif analysis_option == "📌 الجوائز.. هل فعلاً ترفع سعر اللاعب؟":
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        <h2>📌 الجوائز.. هل فعلاً ترفع سعر اللاعب؟ 🏆</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        هل لو اللاعب فاز بجائزة فردية، مثل الكرة الذهبية، بيرتفع سعره؟ ولا ما تفرق؟ 🤨
    </div>
    """, unsafe_allow_html=True)
    
    avg_value_awards = df.groupby('has_award')['current_value'].mean()
    avg_value_awards.index = ["بدون جوائز", "حاصل على جوائز"]
    fig = px.bar(
        x=avg_value_awards.index,
        y=avg_value_awards.values,
        labels={'x': "حالة الجوائز", 'y': "متوسط قيمة الانتقال (€M)"},
        color=avg_value_awards.values,
        color_continuous_scale="teal"
    )
    st.plotly_chart(fig)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        💡 <b>البيانات تبين إنه اللاعبين اللي عندهم جوائز فردية أو ألقاب بارزة يتمتعون بقيمة سوقية أعلى بشكل واضح 
        مقارنةً باللي ما حققوا أي جوائز. الجوائز عادةً تعكس مستوى مميز أو إنجاز كبير، وهذا ينعكس على ثقة الأندية 
        وجماهيرها في اللاعب، ويرفع سعره في سوق الانتقالات. بمعنى آخر، اللاعب الحاصل على جوائز يصير "براند" أقوى، 
        وفي غالب الأحيان تلاقي الأندية تدفع زيادة له، سواءً لأنه يضيف قيمة فنية للفريق أو حتى قيمة تسويقية خارج الملعب.</b>
    </div>
    """, unsafe_allow_html=True)

elif analysis_option == "📌 الإصابات.. مين أكثر مركز يعاني؟":
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        <h2>📌 الإصابات.. مين أكثر مركز يعاني؟ 🤕</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        تخيل إنك تدفع ملايين على لاعب، وبعدها يقعد طول الموسم مصاب.. صفقة خاسرة صح؟ 😬
    </div>
    """, unsafe_allow_html=True)
    
    avg_games_missed_by_position = df.groupby('position_label')['games_injured'].mean().sort_values(ascending=False)
    fig = px.bar(
        x=avg_games_missed_by_position.index,
        y=avg_games_missed_by_position.values,
        labels={'x': "المركز", 'y': "متوسط عدد المباريات المفقودة بسبب الإصابة"},
        color=avg_games_missed_by_position.values,
        color_continuous_scale="teal"
    )
    st.plotly_chart(fig)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        💡 <b>يبيّن الرسم إن المدافعين هم أكثر مركز يتعرّض للإصابات بمعدّل عالي، 
        لأن احتكاكهم المباشر مع المهاجمين ومحاولتهم الدائمة لاستخلاص الكرة يعرّضهم للخشونة بشكل مستمر. 
        يجي بعدهم المهاجمون بنسبة قريبة، وذا مفهوم لأنهم يدخلون في صراعات بدنية خلال الهجوم والاختراق. 
        أما لاعبي الوسط فهم أقل عرضة بقليل نظرًا لتوزيع مهامهم بين الدفاع والهجوم وعدم تركّزهم 
        في تحامات بدنية بنفس وتيرة المدافعين أو المهاجمين. في النهاية، الحراس هم الأقل إصابة 
        بحكم تمركزهم في منطقة واحدة أغلب المباراة، بالرغم من خطورة بعض اللقطات اللي تواجههم أحيانًا.</b>
    </div>
    """, unsafe_allow_html=True)

elif analysis_option == "📌 هل اللاعب يحافظ على قيمته بمرور الوقت؟":
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        <h2>📌 هل اللاعب يحافظ على قيمته بمرور الوقت؟ 📉</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        هل اللاعب اللي سعره مرتفع اليوم، ممكن يظل بنفس السعر بعد 5 سنوات؟ 🤔
    </div>
    """, unsafe_allow_html=True)
    
    fig = px.scatter(
        df, x="highest_value", y="current_value",
        color="value_retention",
        labels={"highest_value": "أعلى قيمة انتقال", "current_value": "القيمة الحالية"},
        color_continuous_scale="teal"
    )
    st.plotly_chart(fig)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        💡 <b>بعض اللاعبين تشوف عندهم فجوة كبيرة بين أعلى قيمة وصلوها والقيمة الحالية، وهذا غالبًا يدل إن مستواهم أو مردودهم 
        في الملعب تراجع بشكل سريع—سواءً بسبب الإصابات أو هبوط الأداء. وفي المقابل، في لاعبين ما كانت قيمتهم الأعلى ضخمة أساسًا، 
        ومع كذا حافظوا على استقرار نسبيّ بين أعلى قيمة وصلوا لها وقيمتهم الحالية، وهذا يوضّح إنهم غالبًا أكثر ثبات 
        أو أقل عرضة للإصابات والصدمات اللي تخلي قيمة اللاعب تنهار فجأة.</b>
    </div>
    """, unsafe_allow_html=True)

elif analysis_option == "🔮 توقع سعر اللاعب":
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        <h2>🔮 توقع سعر اللاعب</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: right; direction: rtl; text-align: justify;">
        أدخل معلومات اللاعب أدناه لتوقع سعر انتقاله:
    </div>
    """, unsafe_allow_html=True)

    position = st.selectbox("اختر مركز اللاعب:", options=["حارس", "مدافع", "لاعب وسط", "مهاجم"], key="position_select")

    with st.form("price_prediction_form"):
        features = {}
        
        if position == "حارس":
            st.markdown("""
            <div style="text-align: right; direction: rtl; text-align: justify;">
                <b>ملء بيانات الحارس</b>
            </div>
            """, unsafe_allow_html=True)
            features["appearance"] = st.number_input("عدد المباريات", min_value=0, value=0, key="gk_appearance_pred")
            features["minutes played"] = st.number_input("دقائق اللعب", min_value=0, value=0, key="gk_minutes_pred")
            features["highest_value"] = st.number_input("أعلى قيمة انتقال سابقة (€M)", min_value=0.0, value=0.0, step=0.1, key="gk_highest_pred")
            features["award"] = 0  # Set awards to 0 by default.
            expected_teams_gk = ["Arsenal FC", "Atlético de Madrid", "FC Porto", "Liverpool FC", "Manchester City", "Paris Saint-Germain", "Real Madrid"]
            team_options_gk = ["Other"] + expected_teams_gk
            team_selected = st.selectbox("اختر الفريق (أو اختر 'Other' إذا كان من فريق آخر):", options=team_options_gk, key="gk_team_pred")
            for team in expected_teams_gk:
                features[f"team_{team}"] = 1 if team == team_selected else 0

            GK_FEATURE_ORDER = ["appearance", "minutes played", "award", "highest_value", 
                                "team_Arsenal FC", "team_Atlético de Madrid", "team_FC Porto", 
                                "team_Liverpool FC", "team_Manchester City", "team_Paris Saint-Germain", "team_Real Madrid"]
            input_features = [features[f] for f in GK_FEATURE_ORDER]
            input_features_scaled = gk_scaler.transform([input_features])
            predicted_price = gk_model.predict(input_features_scaled)[0]

        elif position == "مدافع":
            st.markdown("""
            <div style="text-align: right; direction: rtl; text-align: justify;">
                <b>ملء بيانات المدافع</b>
            </div>
            """, unsafe_allow_html=True)
            features["appearance"] = st.number_input("عدد المباريات", min_value=0, value=0, key="def_appearance_pred")
            features["minutes played"] = st.number_input("دقائق اللعب", min_value=0, value=0, key="def_minutes_pred")
            features["games_injured"] = st.number_input("عدد المباريات المفقودة بسبب الإصابة", min_value=0, value=0, key="def_injured_pred")
            features["highest_value"] = st.number_input("أعلى قيمة انتقال سابقة (€M)", min_value=0.0, value=0.0, step=0.1, key="def_highest_pred")
            features["award"] = 0  # Set awards to 0 by default.
            expected_teams_def = ["Arsenal FC", "Bayern Munich", "Chelsea FC", "FC Barcelona", 
                                  "Liverpool FC", "Manchester City", "Manchester United", "Paris Saint-Germain", "Tottenham Hotspur"]
            team_options_def = ["Other"] + expected_teams_def
            team_selected = st.selectbox("اختر الفريق (أو اختر 'Other' إذا كان من فريق آخر):", options=team_options_def, key="def_team_pred")
            for team in expected_teams_def:
                features[f"team_{team}"] = 1 if team == team_selected else 0

            DEF_FEATURE_ORDER = ["appearance", "minutes played", "games_injured", "award", "highest_value",
                                 "team_Arsenal FC", "team_Bayern Munich", "team_Chelsea FC", "team_FC Barcelona",
                                 "team_Liverpool FC", "team_Manchester City", "team_Manchester United",
                                 "team_Paris Saint-Germain", "team_Tottenham Hotspur"]
            input_features = [features[f] for f in DEF_FEATURE_ORDER]
            input_features_scaled = def_scaler.transform([input_features])
            predicted_price = def_model.predict(input_features_scaled)[0]

        elif position == "لاعب وسط":
            st.markdown("""
            <div style="text-align: right; direction: rtl; text-align: justify;">
                <b>ملء بيانات لاعب الوسط</b>
            </div>
            """, unsafe_allow_html=True)
            features["appearance"] = st.number_input("عدد المباريات", min_value=0, value=0, key="mid_appearance_pred")
            features["minutes played"] = st.number_input("دقائق اللعب", min_value=0, value=0, key="mid_minutes_pred")
            features["highest_value"] = st.number_input("أعلى قيمة انتقال سابقة (€M)", min_value=0.0, value=0.0, step=0.1, key="mid_highest_pred")
            features["award"] = 0  # Set awards to 0 by default.
            expected_teams_mid = ["Arsenal FC", "Bayern Munich", "Chelsea FC", "FC Barcelona", "Manchester City", "Real Madrid"]
            team_options_mid = ["Other"] + expected_teams_mid
            team_selected = st.selectbox("اختر الفريق (أو اختر 'Other' إذا كان من فريق آخر):", options=team_options_mid, key="mid_team_pred")
            for team in expected_teams_mid:
                features[f"team_{team}"] = 1 if team == team_selected else 0

            MID_FEATURE_ORDER = ["appearance", "minutes played", "award", "highest_value",
                                 "team_Arsenal FC", "team_Bayern Munich", "team_Chelsea FC", "team_FC Barcelona",
                                 "team_Manchester City", "team_Real Madrid"]
            input_features = [features[f] for f in MID_FEATURE_ORDER]
            input_features_scaled = mid_scaler.transform([input_features])
            predicted_price = mid_model.predict(input_features_scaled)[0]

        elif position == "مهاجم":
            st.markdown("""
            <div style="text-align: right; direction: rtl; text-align: justify;">
                <b>ملء بيانات المهاجم</b>
            </div>
            """, unsafe_allow_html=True)
            features["appearance"] = st.number_input("عدد المباريات", min_value=0, value=0, key="fw_appearance_pred")
            features["goals"] = st.number_input("الأهداف", min_value=0, value=0, key="fw_goals_pred")
            features["assists"] = st.number_input("التمريرات الحاسمة", min_value=0, value=0, key="fw_assists_pred")
            features["minutes played"] = st.number_input("دقائق اللعب", min_value=0, value=0, key="fw_minutes_pred")
            features["games_injured"] = st.number_input("عدد المباريات المفقودة بسبب الإصابة", min_value=0, value=0, key="fw_injured_pred")
            features["highest_value"] = st.number_input("أعلى قيمة انتقال سابقة (€M)", min_value=0.0, value=0.0, step=0.1, key="fw_highest_pred")
            features["award"] = 0  # Set awards to 0 by default.
            expected_teams_fw = ["Arsenal FC", "Bayern Munich", "Liverpool FC", "Manchester City", 
                                 "Paris Saint-Germain", "Real Madrid", "SSC Napoli", "Tottenham Hotspur"]
            team_options_fw = ["Other"] + expected_teams_fw
            team_selected = st.selectbox("اختر الفريق (أو اختر 'Other' إذا كان من فريق آخر):", options=team_options_fw, key="fw_team_pred")
            for team in expected_teams_fw:
                features[f"team_{team}"] = 1 if team == team_selected else 0

            FW_FEATURE_ORDER = ["appearance", "goals", "assists", "minutes played", "games_injured", "award", "highest_value",
                                "team_Arsenal FC", "team_Bayern Munich", "team_Liverpool FC", "team_Manchester City",
                                "team_Paris Saint-Germain", "team_Real Madrid", "team_SSC Napoli", "team_Tottenham Hotspur"]
            input_features = [features[f] for f in FW_FEATURE_ORDER]
            input_features_scaled = fw_scaler.transform([input_features])
            predicted_price = fw_model.predict(input_features_scaled)[0]

        submit = st.form_submit_button("توقع سعر اللاعب")
        if submit:
            predicted_price = abs(predicted_price)  # If the predicted price is negative, take its absolute value.
            formatted_price = "{:,.0f}".format(round(predicted_price))
            st.markdown(f"""
            <div style="text-align: right; direction: rtl; text-align: justify;">
                <span style="color: green; font-size: 18px;">
                    السعر المتوقع: {formatted_price} يورو
                </span>
            </div>
            """, unsafe_allow_html=True)
