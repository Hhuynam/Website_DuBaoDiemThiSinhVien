# train_model.py
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression

def train_model(file_path, model_type):
    """
    Huấn luyện mô hình dự báo dựa trên file CSV và kiểu mô hình chọn (regression hoặc classification)
    
    Tham số:
      file_path (str): Đường dẫn file CSV upload
      model_type (str): 'regression' hoặc 'classification'
    
    Trả về:
      dict với:
        "table_html": bảng kết quả dự báo 10 dòng đầu, được chuyển sang HTML string
        "plot_base64": chuỗi Base64 của ảnh biểu đồ được tạo bằng matplotlib
    """
    # Đọc file CSV
    df = pd.read_csv(file_path)

    # Giả sử file CSV có cột "math score". Tạo thêm cột pass_fail: nếu >= 60 thì là 1, ngược lại là 0
    passing_threshold = 60
    df['pass_fail'] = df['math score'].apply(lambda x: 1 if x >= passing_threshold else 0)
    
    # Các đặc trưng dùng để dự báo
    features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    X = df[features]
    
    # Chọn biến mục tiêu theo kiểu mô hình
    if model_type == 'regression':
        y = df['math score']
    else:
        y = df['pass_fail']
        
    # Khởi tạo bộ tiền xử lý: OneHotEncoder cho các biến categorical
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, features)
    ], remainder='passthrough')
    
    # Chọn mô hình phù hợp
    if model_type == 'regression':
        model = LinearRegression()
    else:
        model = LogisticRegression(max_iter=200)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Huấn luyện mô hình
    pipeline.fit(X, y)
    
    # Dự đoán trên toàn bộ dữ liệu
    predictions = pipeline.predict(X)
    
    # Tạo bảng kết quả với 10 dòng đầu
    result_df = df.copy()
    if model_type == 'regression':
        result_df['predicted_score'] = predictions
        display_columns = ['math score', 'predicted_score']
    else:
        result_df['predicted_pass_fail'] = predictions.astype(int)
        display_columns = ['pass_fail', 'predicted_pass_fail']
        
    table_html = result_df[display_columns].head(10).to_html(classes="table", index=False)
    
    # Tạo biểu đồ bằng matplotlib
    plt.figure(figsize=(8, 6))
    if model_type == 'regression':
        # Vẽ scatter plot so sánh giá trị thực và dự đoán
        plt.scatter(df['math score'], result_df['predicted_score'], color='blue', label='Predicted vs Actual')
        plt.plot([df['math score'].min(), df['math score'].max()],
                 [df['math score'].min(), df['math score'].max()],
                 color='red', linestyle='--', label='Ideal')
        plt.xlabel('Actual Math Score')
        plt.ylabel('Predicted Math Score')
        plt.legend()
        plt.title('Regression: Actual vs Predicted Math Score')
    else:
        # Vẽ bar chart: phân bố số lượng dự đoán Pass/Fail
        counts = result_df['predicted_pass_fail'].value_counts()
        labels = ['Failed', 'Passed']
        values = [counts.get(0, 0), counts.get(1, 0)]
        plt.bar(labels, values, color=['orange', 'green'])
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.title('Classification: Predicted Pass/Fail Distribution')
    
    # Lưu ảnh vào bộ nhớ và chuyển qua chuỗi Base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return {
        "table_html": table_html,
        "plot_base64": plot_base64
    }
