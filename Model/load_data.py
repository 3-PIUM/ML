from sklearn.datasets import load_iris
import pandas as pd

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data
feature_names = [
    "sepal_length_cm", "sepal_width_cm",
    "petal_length_cm", "petal_width_cm"
]

# DataFrame으로 변환
df = pd.DataFrame(X, columns=feature_names)

# 저장
df.to_csv("iris_test_data.csv", index=False)
print("✅ iris_test_data.csv 생성 완료")
