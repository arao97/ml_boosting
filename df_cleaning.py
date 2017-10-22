from sklearn import preprocessing


def transform_attributes(df, attributes):
    label_encoder = preprocessing.LabelEncoder()
    for attribute in attributes:
        df[str(attribute)] = label_encoder.fit_transform(df[str(attribute)])

    return df


def preprocessor(df):
    new_df = df.copy()
    new_df = transform_attributes(new_df,
                                  ['"job"', '"marital"',
                                   '"education"', '"default"',
                                   '"housing"', '"month"',
                                   '"loan"', '"contact"',
                                   '"day_of_week"', '"poutcome"',
                                   '"y"'])
    return new_df
