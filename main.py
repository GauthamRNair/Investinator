from model import Investinator
import keras.models

if __name__ == "__main__":
    if_train = input("Train? (y/n): ").lower()
    investinator = Investinator("COST")
    if if_train == "y":
        investinator.train()
        investinator.model.save("COST.keras")
        investinator.predict()
    else:
        investinator.train(keras.models.load_model("COST.keras"))
        investinator.predict()
