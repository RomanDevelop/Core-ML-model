import coremltools as ct
from PIL import Image

# Загрузка модели
model_path = '/Users/anymacstore/Documents/MyImageClassifier.mlmodel'
model = ct.models.MLModel(model_path)

# Метаданные модели
metadata = {
    "model_name": model.short_description,
    "input_description": model.input_description,
    "output_description": model.output_description,
    "input_features": model.get_spec().description.input,
    "output_features": model.get_spec().description.output,
}
print("Метаданные модели:")
for key, value in metadata.items():
    print(f"{key}: {value}")

# Функция для загрузки и подготовки изображения
def preprocess_image(image_path, target_size):
    image = Image.open(image_path).resize(target_size)
    return image  # Возвращаем объект PIL.Image.Image

# Путь к изображению и размер
test_image_path = '/Users/anymacstore/Downloads/test_image.jpg'
input_size = (224, 224)
processed_image = preprocess_image(test_image_path, input_size)

# Получение названия входного параметра
input_name = model.get_spec().description.input[0].name

# Прогноз модели
prediction = model.predict({input_name: processed_image})

# Результат классификации
print("Результат классификации:", prediction)
