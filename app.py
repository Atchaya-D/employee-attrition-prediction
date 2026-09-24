import streamlit as st
import pandas as pd
import joblib
import os


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load("best_attrition_model.pkl")


# --------------------------------------------------
# Load Dataset for Dashboard
# --------------------------------------------------

data_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = None


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Employee Attrition Prediction System")

st.write(
    "Predict employee attrition risk using machine learning."
)


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page:",
    [
        "Dashboard",
        "Attrition Prediction"
    ]
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header("HR Analytics Dashboard")

    if df is None:

        st.info(
            "Dashboard data is not available in the deployed version. "
            "The Attrition Prediction page is available below."
        )

        st.subheader("About this project")

        st.write(
            "This application uses a Random Forest machine learning "
            "model to predict employee attrition based on employee "
            "and job-related information."
        )

        st.write(
            "Please use the 'Attrition Prediction' page from the "
            "sidebar to make a prediction."
        )

    else:

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

        # ------------------------------------------
        # Attrition by Department
        # ------------------------------------------

        st.subheader("Attrition by Department")

        department_attrition = pd.crosstab(
            df["Department"],
            df["Attrition"]
        )

        st.bar_chart(department_attrition)

        # ------------------------------------------
        # Attrition by Job Role
        # ------------------------------------------

        st.subheader("Attrition by Job Role")

        role_attrition = pd.crosstab(
            df["JobRole"],
            df["Attrition"]
        )

        st.bar_chart(role_attrition)

        # ------------------------------------------
        # Attrition by Overtime
        # ------------------------------------------

        st.subheader("Attrition by Overtime")

        overtime_attrition = pd.crosstab(
            df["OverTime"],
            df["Attrition"]
        )

        st.bar_chart(overtime_attrition)


# ==================================================
# ATTRITION PREDICTION
# ==================================================

if page == "Attrition Prediction":

    st.header("🔮 Employee Attrition Prediction")

    st.write(
        "Enter employee details to predict the probability "
        "of employee attrition."
    )

    st.subheader("Employee Information")

    # --------------------------------------------------
    # Column 1
    # --------------------------------------------------

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
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel"
            ]
        )

        daily_rate = st.number_input(
            "Daily Rate",
            min_value=102,
            max_value=1499,
            value=800
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
            max_value=29,
            value=5
        )

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

        environment_satisfaction = st.number_input(
            "Environment Satisfaction",
            min_value=1,
            max_value=4,
            value=3
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            min_value=30,
            max_value=100,
            value=65
        )


    # --------------------------------------------------
    # Column 2
    # --------------------------------------------------

    with col2:

        job_involvement = st.number_input(
            "Job Involvement",
            min_value=1,
            max_value=4,
            value=3
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

        job_satisfaction = st.number_input(
            "Job Satisfaction",
            min_value=1,
            max_value=4,
            value=3
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced"
            ]
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1009,
            max_value=19999,
            value=5000
        )

        monthly_rate = st.number_input(
            "Monthly Rate",
            min_value=2094,
            max_value=26999,
            value=14000
        )

        num_companies = st.number_input(
            "Number of Companies Worked",
            min_value=0,
            max_value=9,
            value=2
        )

        overtime = st.selectbox(
            "OverTime",
            [
                "Yes",
                "No"
            ]
        )

        percent_salary_hike = st.number_input(
            "Percent Salary Hike",
            min_value=11,
            max_value=25,
            value=15
        )


    # --------------------------------------------------
    # Column 3
    # --------------------------------------------------

    with col3:

        performance_rating = st.number_input(
            "Performance Rating",
            min_value=3,
            max_value=4,
            value=3
        )

        relationship_satisfaction = st.number_input(
            "Relationship Satisfaction",
            min_value=1,
            max_value=4,
            value=3
        )

        stock_option_level = st.number_input(
            "Stock Option Level",
            min_value=0,
            max_value=3,
            value=1
        )

        total_working_years = st.number_input(
            "Total Working Years",
            min_value=0,
            max_value=40,
            value=8
        )

        training_times = st.number_input(
            "Training Times Last Year",
            min_value=0,
            max_value=6,
            value=3
        )

        work_life_balance = st.number_input(
            "Work Life Balance",
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

        years_current_role = st.number_input(
            "Years In Current Role",
            min_value=0,
            max_value=18,
            value=3
        )

        years_promotion = st.number_input(
            "Years Since Last Promotion",
            min_value=0,
            max_value=15,
            value=2
        )

        years_manager = st.number_input(
            "Years With Current Manager",
            min_value=0,
            max_value=17,
            value=3
        )


    # --------------------------------------------------
    # Prediction Button
    # --------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🔮 Predict Attrition Risk",
        type="primary"
    )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    if predict_button:

        employee = pd.DataFrame({

            "Age": [age],

            "DailyRate": [daily_rate],

            "DistanceFromHome": [distance_from_home],

            "Education": [education],

            "EnvironmentSatisfaction": [
                environment_satisfaction
            ],

            "HourlyRate": [hourly_rate],

            "JobInvolvement": [
                job_involvement
            ],

            "JobLevel": [job_level],

            "JobSatisfaction": [
                job_satisfaction
            ],

            "MonthlyIncome": [
                monthly_income
            ],

            "MonthlyRate": [
                monthly_rate
            ],

            "NumCompaniesWorked": [
                num_companies
            ],

            "PercentSalaryHike": [
                percent_salary_hike
            ],

            "PerformanceRating": [
                performance_rating
            ],

            "RelationshipSatisfaction": [
                relationship_satisfaction
            ],

            "StockOptionLevel": [
                stock_option_level
            ],

            "TotalWorkingYears": [
                total_working_years
            ],

            "TrainingTimesLastYear": [
                training_times
            ],

            "WorkLifeBalance": [
                work_life_balance
            ],

            "YearsAtCompany": [
                years_at_company
            ],

            "YearsInCurrentRole": [
                years_current_role
            ],

            "YearsSinceLastPromotion": [
                years_promotion
            ],

            "YearsWithCurrManager": [
                years_manager
            ],

            "BusinessTravel": [
                business_travel
            ],

            "Department": [
                department
            ],

            "EducationField": [
                education_field
            ],

            "Gender": [
                gender
            ],

            "JobRole": [
                job_role
            ],

            "MaritalStatus": [
                marital_status
            ],

            "OverTime": [
                overtime
            ]
        })


        # ------------------------------------------
        # Model Prediction
        # ------------------------------------------

        prediction = model.predict(employee)[0]

        probability = model.predict_proba(
            employee
        )[0][1]


        # ------------------------------------------
        # Result
        # ------------------------------------------

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ High Attrition Risk"
            )

        else:

            st.success(
                "✅ Low Attrition Risk"
            )

        st.metric(
            "Attrition Probability",
            f"{probability:.2%}"
        )
        