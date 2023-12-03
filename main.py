from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from model import Investinator
import keras.models


if __name__ == "__main__":
    stock_name = input("Stock name: ").upper()
    if_train = input("Train? (y/n): ").lower()
    investinator = Investinator(stock_name)
    if if_train == "y":
        period = input("Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max): ")
        investinator.train(period=period)
        investinator.model.save("models/" + stock_name + ".keras")
        investinator.predict()
    else:
        investinator.train(keras.models.load_model("models/" + stock_name + ".keras"))
        investinator.predict()
