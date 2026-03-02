def add_bmi_category(df):
    df['BMI_Category'] = df['BMI'].apply(
        lambda x: 1 if x < 25 else 2 if x < 30 else 3
    )
    return df