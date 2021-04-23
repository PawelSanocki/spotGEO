from model.create_dataset import create_dataset
from model.train_model import get_trained_model
from runner_NN import run
print()
print("Creating dataset")
create_dataset()
print()
print("Training")
model = get_trained_model()
print("Running")
run(model)


