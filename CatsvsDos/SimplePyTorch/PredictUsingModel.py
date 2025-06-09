
from SimpleCNN import  SimpleCNN

done = False

model = SimpleCNN()
model.load_model()

while (done == False):
    image_file = input("Enter the path of the image file. To exist, enter exit:")
    if image_file == 'exit':
        break
    else:
        print(f"Prediction: {model.predict_image(image_file)}")
        
