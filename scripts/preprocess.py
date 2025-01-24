from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_categorical(data, columns):
    """Encode categorical variables."""
    encoder = LabelEncoder()
    for col in columns:
        data[col] = encoder.fit_transform(data[col])
    return data

def scale_features(data, columns):
    """Scale numerical features."""
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data
