import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load("best_attrition_model.pkl")
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction System")
st.write(
    "Predict employee attrition risk using machine learning."
)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page:",
    [
        "Dashboard",
        "Attrition Prediction"
    ]
)
df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

if page == "Dashboard":

    st.header("HR Analytics Dashboard")

    total_employees = len(df)

    attrition_count = (
        df["Attrition"] == "Yes"
    ).sum()

    attrition_rate = (
        attrition_count / total_employees
    ) * 100
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Employees",
        total_employees
    )

    col2.metric(
        "Employees Left",
        attrition_count
    )

    col3.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )
    st.subheader("Attrition by Department")

    department_attrition = pd.crosstab(
        df["Department"],
        df["Attrition"]
    )

    st.bar_chart(department_attrition)
    st.subheader("Attrition by Job Role")

    role_attrition = pd.crosstab(
        df["JobRole"],
        df["Attrition"]
    )

    st.bar_chart(role_attrition)
    st.subheader("Attrition by Overtime")

    overtime_attrition = pd.crosstab(
        df["OverTime"],
        df["Attrition"]
    )

    st.bar_chart(overtime_attrition)
if page == "Attrition Prediction":

    st.header("🔮 Employee Attrition Prediction")
    st.write("Enter employee details to predict the probability of attrition.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=60,
            value=30
        )

        business_travel = st.selectbox(
            "Business Travel",
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        )

        department = st.selectbox(
            "Department",
            [
                "Sales",
                "Research & Development",
                "Human Resources"
            ]
        )

        distance_from_home = st.number_input(
            "Distance From Home",
            min_value=1,
            max_value=30,
            value=5
        )

    with col2:
        education = st.number_input(
            "Education",
            min_value=1,
            max_value=5,
            value=3
        )

        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other"
            ]
        )

        job_level = st.number_input(
            "Job Level",
            min_value=1,
            max_value=5,
            value=2
        )

        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ]
        )

    with col3:
        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            max_value=20000,
            value=5000
        )

        overtime = st.selectbox(
            "OverTime",
            ["Yes", "No"]
        )

        job_satisfaction = st.number_input(
            "Job Satisfaction",
            min_value=1,
            max_value=4,
            value=3
        )

        years_at_company = st.number_input(
            "Years At Company",
            min_value=0,
            max_value=40,
            value=5
        )

    predict_button = st.button(
        "🔮 Predict Attrition Risk"
    )

    if predict_button:

        employee = pd.DataFrame({
            "Age": [age],
            "BusinessTravel": [business_travel],
            "Department": [department],
            "DistanceFromHome": [distance_from_home],
            "Education": [education],
            "EducationField": [education_field],
            "JobLevel": [job_level],
            "JobRole": [job_role],
            "MonthlyIncome": [monthly_income],
            "OverTime": [overtime],
            "JobSatisfaction": [job_satisfaction],
            "YearsAtCompany": [years_at_company]
        })

        prediction = model.predict(employee)[0]
        probability = model.predict_proba(employee)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Attrition Risk")
        else:
            st.success("✅ Low Attrition Risk")

        st.metric(
            "Attrition Probability",
            f"{probability:.2%}"
        )
if page == "Attrition Prediction":

    st.header("🔮 Employee Attrition Prediction")
    st.write(
        "Enter employee details to predict the probability of employee attrition."
    )

    st.subheader("Employee Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=int(df["Age"].min()),
            max_value=int(df["Age"].max()),
            value=int(df["Age"].median())
        )

        business_travel = st.selectbox(
            "Business Travel",
            df["BusinessTravel"].unique()
        )

        daily_rate = st.number_input(
            "Daily Rate",
            min_value=int(df["DailyRate"].min()),
            max_value=int(df["DailyRate"].max()),
            value=int(df["DailyRate"].median())
        )

        department = st.selectbox(
            "Department",
            df["Department"].unique()
        )

        distance_from_home = st.number_input(
            "Distance From Home",
            min_value=int(df["DistanceFromHome"].min()),
            max_value=int(df["DistanceFromHome"].max()),
            value=int(df["DistanceFromHome"].median())
        )

        education = st.number_input(
            "Education",
            min_value=int(df["Education"].min()),
            max_value=int(df["Education"].max()),
            value=int(df["Education"].median())
        )

        education_field = st.selectbox(
            "Education Field",
            df["EducationField"].unique()
        )

        environment_satisfaction = st.number_input(
            "Environment Satisfaction",
            min_value=int(df["EnvironmentSatisfaction"].min()),
            max_value=int(df["EnvironmentSatisfaction"].max()),
            value=int(df["EnvironmentSatisfaction"].median())
        )

        gender = st.selectbox(
            "Gender",
            df["Gender"].unique()
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            min_value=int(df["HourlyRate"].min()),
            max_value=int(df["HourlyRate"].max()),
            value=int(df["HourlyRate"].median())
        )

    with col2:

        job_involvement = st.number_input(
            "Job Involvement",
            min_value=int(df["JobInvolvement"].min()),
            max_value=int(df["JobInvolvement"].max()),
            value=int(df["JobInvolvement"].median())
        )

        job_level = st.number_input(
            "Job Level",
            min_value=int(df["JobLevel"].min()),
            max_value=int(df["JobLevel"].max()),
            value=int(df["JobLevel"].median())
        )

        job_role = st.selectbox(
            "Job Role",
            df["JobRole"].unique()
        )

        job_satisfaction = st.number_input(
            "Job Satisfaction",
            min_value=int(df["JobSatisfaction"].min()),
            max_value=int(df["JobSatisfaction"].max()),
            value=int(df["JobSatisfaction"].median())
        )

        marital_status = st.selectbox(
            "Marital Status",
            df["MaritalStatus"].unique()
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=int(df["MonthlyIncome"].min()),
            max_value=int(df["MonthlyIncome"].max()),
            value=int(df["MonthlyIncome"].median())
        )

        monthly_rate = st.number_input(
            "Monthly Rate",
            min_value=int(df["MonthlyRate"].min()),
            max_value=int(df["MonthlyRate"].max()),
            value=int(df["MonthlyRate"].median())
        )

        num_companies = st.number_input(
            "Number of Companies Worked",
            min_value=int(df["NumCompaniesWorked"].min()),
            max_value=int(df["NumCompaniesWorked"].max()),
            value=int(df["NumCompaniesWorked"].median())
        )

        overtime = st.selectbox(
            "OverTime",
            df["OverTime"].unique()
        )

        percent_salary_hike = st.number_input(
            "Percent Salary Hike",
            min_value=int(df["PercentSalaryHike"].min()),
            max_value=int(df["PercentSalaryHike"].max()),
            value=int(df["PercentSalaryHike"].median())
        )

    with col3:

        performance_rating = st.number_input(
            "Performance Rating",
            min_value=int(df["PerformanceRating"].min()),
            max_value=int(df["PerformanceRating"].max()),
            value=int(df["PerformanceRating"].median())
        )

        relationship_satisfaction = st.number_input(
            "Relationship Satisfaction",
            min_value=int(df["RelationshipSatisfaction"].min()),
            max_value=int(df["RelationshipSatisfaction"].max()),
            value=int(df["RelationshipSatisfaction"].median())
        )

        stock_option_level = st.number_input(
            "Stock Option Level",
            min_value=int(df["StockOptionLevel"].min()),
            max_value=int(df["StockOptionLevel"].max()),
            value=int(df["StockOptionLevel"].median())
        )

        total_working_years = st.number_input(
            "Total Working Years",
            min_value=int(df["TotalWorkingYears"].min()),
            max_value=int(df["TotalWorkingYears"].max()),
            value=int(df["TotalWorkingYears"].median())
        )

        training_times = st.number_input(
            "Training Times Last Year",
            min_value=int(df["TrainingTimesLastYear"].min()),
            max_value=int(df["TrainingTimesLastYear"].max()),
            value=int(df["TrainingTimesLastYear"].median())
        )

        work_life_balance = st.number_input(
            "Work Life Balance",
            min_value=int(df["WorkLifeBalance"].min()),
            max_value=int(df["WorkLifeBalance"].max()),
            value=int(df["WorkLifeBalance"].median())
        )

        years_at_company = st.number_input(
            "Years At Company",
            min_value=int(df["YearsAtCompany"].min()),
            max_value=int(df["YearsAtCompany"].max()),
            value=int(df["YearsAtCompany"].median())
        )

        years_current_role = st.number_input(
            "Years In Current Role",
            min_value=int(df["YearsInCurrentRole"].min()),
            max_value=int(df["YearsInCurrentRole"].max()),
            value=int(df["YearsInCurrentRole"].median())
        )

        years_promotion = st.number_input(
            "Years Since Last Promotion",
            min_value=int(df["YearsSinceLastPromotion"].min()),
            max_value=int(df["YearsSinceLastPromotion"].max()),
            value=int(df["YearsSinceLastPromotion"].median())
        )

        years_manager = st.number_input(
            "Years With Current Manager",
            min_value=int(df["YearsWithCurrManager"].min()),
            max_value=int(df["YearsWithCurrManager"].max()),
            value=int(df["YearsWithCurrManager"].median())
        )

    st.divider()

    predict_button = st.button(
        "🔮 Predict Attrition Risk",
        type="primary"
    )

    if predict_button:

        employee = pd.DataFrame({
            "Age": [age],
            "DailyRate": [daily_rate],
            "DistanceFromHome": [distance_from_home],
            "Education": [education],
            "EnvironmentSatisfaction": [environment_satisfaction],
            "HourlyRate": [hourly_rate],
            "JobInvolvement": [job_involvement],
            "JobLevel": [job_level],
            "JobSatisfaction": [job_satisfaction],
            "MonthlyIncome": [monthly_income],
            "MonthlyRate": [monthly_rate],
            "NumCompaniesWorked": [num_companies],
            "PercentSalaryHike": [percent_salary_hike],
            "PerformanceRating": [performance_rating],
            "RelationshipSatisfaction": [relationship_satisfaction],
            "StockOptionLevel": [stock_option_level],
            "TotalWorkingYears": [total_working_years],
            "TrainingTimesLastYear": [training_times],
            "WorkLifeBalance": [work_life_balance],
            "YearsAtCompany": [years_at_company],
            "YearsInCurrentRole": [years_current_role],
            "YearsSinceLastPromotion": [years_promotion],
            "YearsWithCurrManager": [years_manager],

            "BusinessTravel": [business_travel],
            "Department": [department],
            "EducationField": [education_field],
            "Gender": [gender],
            "JobRole": [job_role],
            "MaritalStatus": [marital_status],
            "OverTime": [overtime]
        })

        prediction = model.predict(employee)[0]

        probability = model.predict_proba(employee)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ High Attrition Risk")

        else:

            st.success("✅ Low Attrition Risk")

        st.metric(
            "Attrition Probability",
            f"{probability:.2%}"
        )