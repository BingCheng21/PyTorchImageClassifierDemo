from CatsvsDos.SimplePyTorch.SimpleCNN import *

loaders, class_names = get_data_loaders()

model = SimpleCNN()
train_model(model, loaders, epochs=3)
evaluate_model(model, loaders['val'])
save_model(model, class_names)

