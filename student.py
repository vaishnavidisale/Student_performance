import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------- TITLE & HEADER ---------------------
st.title("Student Performance Analysis")

st.markdown("""
    <div style='background-color:#f0f0f5; padding:20px; border-radius:10px;'>
        <h2 style='color:#003399;'>🎓 Student Performance Dashboard</h2>
        <p>Track progress and get smart learning paths</p>
    </div>
""", unsafe_allow_html=True)

# --------------------- UPLOAD GUIDE ---------------------
st.markdown("""
<div style='background-color:#f0f0f5; padding:15px; border-radius:10px;'>
<h3 style='color:#003399;'>📄 CSV Upload Instructions</h3>
<p>To use the dashboard, your CSV file must contain the following columns:</p>
</div>
""", unsafe_allow_html=True)

# Column Guide Table
column_guide = pd.DataFrame({
    "Column Name": ["StudentID", "Name", "Gender", "AttendanceRate", "StudyHours", 
                    "PreviousGrade", "subject", "exam_type", "marks"],
    "Data Type / Format": ["STRING", "STRING", "STRING", "FLOAT (0–100)", "FLOAT", 
                           "FLOAT", "STRING", "STRING", "FLOAT (0–100)"],
    "Description / Example": ["Unique ID, e.g., S001", 
                              "Student name, e.g., Tanvi Shah", 
                              "Male / Female / Other", 
                              "Attendance %, e.g., 85.5", 
                              "Avg. study hours per week, e.g., 4.5", 
                              "Previous grade, e.g., 67", 
                              "Subject name, e.g., Mathematics", 
                              "Exam type, e.g., Unit Test", 
                              "Marks obtained, e.g., 78"]
})
st.table(column_guide)

# --------------------- FILE UPLOAD ---------------------
data = st.file_uploader("📂 Upload file (csv/xlsx/json/pkl)", type=["csv", "xlsx", "json", "pkl"])

if data is not None:
    # Read file
    if data.name.endswith('.csv'):
        df = pd.read_csv(data)
    elif data.name.endswith('.xlsx'):
        df = pd.read_excel(data)
    elif data.name.endswith('.json'):
        df = pd.read_json(data)
    elif data.name.endswith('.pkl'):
        df = pd.read_pickle(data)
    
    st.dataframe(df)
    st.info("✅ File uploaded successfully!")

    # --------------------- TABS ---------------------
    tab1, tab2, tab3, tab4, tab5 ,tab6= st.tabs([
        "Student Profile Summary", 
        "AI Performance Summary",
        "Subject Insights", 
        "Attendance Analysis", 
        "Comparison Analysis", 
        "Performance Prediction"
    ])

    # --------------------- TAB 1: STUDENT PROFILE ---------------------
    with tab2:
        st.subheader("🧠 AI Smart Performance Summary")
        st.markdown("<hr>", unsafe_allow_html=True)

        avg_marks = df['marks'].mean() if 'marks' in df.columns else df['PreviousGrade'].mean()
        top_students = df[df['marks'] > avg_marks + 10] if 'marks' in df.columns else df[df['PreviousGrade'] > avg_marks + 10]
        low_students = df[df['marks'] < avg_marks - 10] if 'marks' in df.columns else df[df['PreviousGrade'] < avg_marks - 10]

        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Class Average", f"{avg_marks:.2f}")
        col2.metric("🏆 Top Performers", len(top_students))
        col3.metric("⚠️ Students Needing Help", len(low_students))

        st.write("### 🏅 Top 5 Students by Performance")
        top5 = df.sort_values('marks' if 'marks' in df.columns else 'PreviousGrade', ascending=False).head(5)
        st.dataframe(top5[['Name', 'subject', 'marks' if 'marks' in df.columns else 'PreviousGrade']])

        st.write("### 🚨 Students Who Need Attention")
        low5 = df.sort_values('marks' if 'marks' in df.columns else 'PreviousGrade', ascending=True).head(5)
        st.dataframe(low5[['Name', 'subject', 'marks' if 'marks' in df.columns else 'PreviousGrade']])
    with tab1:
        st.subheader("📋 Student Profile Summary")
        st.write(f"**Total Students:** {df.shape[0]}")
        st.write(f"**Total Features:** {df.shape[1]}")
        st.write("**Features:**", list(df.columns))
        st.dataframe(df.describe())

        st.markdown("""
          <style>
           .big-font {
             font-size:30px !important;
             color: darkblue;
            }
          </style>
          <p class="big-font">🔥 Smart Learning Path Recommendation</p>
        """, unsafe_allow_html=True)
    
        student_name = st.selectbox("Select Student", df["Name"].values)
        student_row = df[df["Name"] == student_name].iloc[0]
        Grade = student_row['PreviousGrade']

        def recommend_topic(Grade):
            if Grade < 50:
                return "❗ Revise the topic."
            elif 50 <= Grade < 70:
                return "⚠️ Optional revision recommended."
            else:
                return "✅ Proceed to the next subject."
     
        recommended = recommend_topic(Grade)
        st.success(f"🎯 Recommended Learning Path: **{recommended}**")

    # --------------------- TAB 2: SUBJECT INSIGHTS ---------------------
    with tab3:
        st.subheader("📘 Subject Difficulty & Strength Finder")
        st.markdown("<hr>", unsafe_allow_html=True)

        if 'marks' in df.columns:
            subject_avg = df.groupby('subject')['marks'].mean().sort_values(ascending=False)
        else:
            subject_avg = df.groupby('subject')['PreviousGrade'].mean().sort_values(ascending=False)

        st.write("### 🔝 Subject Performance Overview")
        fig_sub = px.bar(
            subject_avg,
            x=subject_avg.index,
            y=subject_avg.values,
            color=subject_avg.values,
            color_continuous_scale='Blues',
            labels={'x': 'Subject', 'y': 'Average Marks'},
            title="Average Marks per Subject"
        )
        st.plotly_chart(fig_sub, use_container_width=True)

        # best_subject = subject_avg.idxmax()
        # print(best_subject)
        # worst_subject = subject_avg.idxmin()
        # print(worst_subject)

        # st.success(f"🏆 **Strongest Subject:** {best_subject} — Avg: {subject_avg.max():.2f}")
        # st.error(f"⚠️ **Weakest Subject:** {worst_subject} — Avg: {subject_avg.min():.2f}")

        # st.info("💡 Insight: Focus on improving the weakest subject to boost class average.")

        # st.markdown("<hr>", unsafe_allow_html=True)
        # st.subheader("🚨 Automatic Risk Alert System")

        # Risk detection
        # risk_df = df[
        #     (df['AttendanceRate'] < 60) | 
        #     (df['PreviousGrade'] < 50) |
        #     (df['StudyHours'] < 3)
        # ][['Name', 'AttendanceRate', 'PreviousGrade', 'StudyHours']]

        # if risk_df.empty:
        #     st.success("✅ No students currently at risk — great job!")
        # else:
        #     st.error(f"🚨 {len(risk_df)} students are flagged as at risk.")
        #     st.dataframe(risk_df)

            # df['RiskScore'] = (
            #     (100 - df['AttendanceRate']) * 0.4 +
            #     (100 - df['PreviousGrade']) * 0.4 +
            #     (5 - df['StudyHours']) * 4
            # )
            # df['RiskLevel'] = pd.cut(
            #     df['RiskScore'], bins=[0, 50, 100, 150, 200],
            #     labels=['Low', 'Moderate', 'High', 'Critical']
            # )

            # fig_risk = px.bar(
            #     df.sort_values('RiskScore', ascending=False),
            #     x='Name', y='RiskScore',
            #     title="Overall Student Risk Score",
            #     labels={'RiskScore': 'Risk Score', 'Name': 'Student'}
            # )
            # st.plotly_chart(fig_risk, use_container_width=True)
            # st.info("📊 Risk Guide: Low → Stable | Moderate → Needs Attention | High/Critical → Urgent Support Needed")

    # --------------------- TAB 3: ATTENDANCE ---------------------
    with tab4:
        st.subheader("📈 Attendance vs Performance Analysis")
        col1, col2 = st.columns(2)
        col1.metric("📉 Avg. Attendance", f"{df['AttendanceRate'].mean():.1f}%")
        col2.metric("📈 Avg. PreviousGrade", f"{df['PreviousGrade'].mean():.1f}")

        st.info("Attendance Rate vs PreviousGrade", icon="📊")
        fig1 = px.scatter(df, x="AttendanceRate", y="PreviousGrade", size="PreviousGrade", hover_name="Name")
        st.plotly_chart(fig1, use_container_width=True)

        df["Attendance Level"] = pd.cut(df["AttendanceRate"],
                                        bins=[0, 60, 80, 100],
                                        labels=["Low (<60%)", "Medium (60-80%)", "High (>80%)"])

        att_dist = df["Attendance Level"].value_counts().sort_index()
        st.info("Student Attendance Distribution")
        fig2 = px.bar(att_dist, x=att_dist.index, y=att_dist.values,
                      labels={'x': "Attendance Category", 'y': "Number of Students"},
                      color=att_dist.index)
        st.plotly_chart(fig2, use_container_width=True)

        st.info("🚨 Students with Low Attendance (<60%)")
        low_att = df[df["AttendanceRate"] < 60][["Name", "AttendanceRate", "PreviousGrade"]]
        st.dataframe(low_att, use_container_width=True)

    # --------------------- TAB 4: COMPARISON ---------------------
    with tab5:
        st.subheader("📊 Subject Comparison Analysis")
        student_name = st.selectbox("Select Student", df["Name"].unique(), key="student_select")
        student_df = df[df["Name"] == student_name]

        subject_name = st.selectbox("Select Subject", student_df["subject"].unique(), key="subject_select")
        subject_df = student_df[student_df["subject"] == subject_name]

        if subject_df.empty:
            st.warning("No data available for this student and subject.")
        else:
            exam_order = ["Unit Test", "Midterm", "Final"]
            subject_df['exam_type'] = pd.Categorical(subject_df['exam_type'], categories=exam_order, ordered=True)
            subject_df = subject_df.sort_values('exam_type')

            fig = px.line(subject_df, x='exam_type', y='PreviousGrade',
                          markers=True,
                          title=f"{student_name} - {subject_name} Performance Over Exams",
                          labels={'exam_type': 'Exam', 'PreviousGrade': 'Grade'})
            st.plotly_chart(fig, use_container_width=True)

    # --------------------- TAB 5: PREDICTION ---------------------
    with tab6:
        st.subheader("🎯 Performance Prediction")
        np.random.seed(42)
        df_train = df.copy()

        df_train['marks'] = np.clip(
            50 + df_train['AttendanceRate']*0.3 + df_train['PreviousGrade']*0.4 + df_train['StudyHours']*3 + np.random.normal(0,5,len(df_train)),
            0, 100
        )

        model = LinearRegression()
        model.fit(df_train[['AttendanceRate','PreviousGrade','StudyHours']], df_train['marks'])

        student_name = st.selectbox("Select Student", df['Name'].unique(), key="pred_student")
        student_row = df[df['Name']==student_name].iloc[0]

        predicted_score = model.predict([[student_row['AttendanceRate'], student_row['PreviousGrade'], student_row['StudyHours']]])[0]
        predicted_score = max(0, min(predicted_score, 100))

        st.success(f"📊 Predicted Final Exam Marks for {student_name}: **{predicted_score:.2f}**")
