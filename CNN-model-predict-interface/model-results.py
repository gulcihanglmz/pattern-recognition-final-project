from tensorflow.keras.models import load_model

# Modeli yükle
model = load_model('D:/pycharm-project/pythonProject/fruit_model.h5')

# Modelin bir özetini yazdırarak doğru yüklendiğini kontrol et
model.summary()
