import graph as generator
from tensorflow.python import keras


def createTriplet(filters=112, stride=1):
    _block = keras.Sequential([
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(3, strides=stride),
        keras.layers.Conv2D(filters, 1),
        keras.layers.BatchNormalization()
    ], "triplet")
    return _block


def createNetwork(graph, input_shape):
    model = [
        keras.layers.InputLayer(input_shape),
        createTriplet()
    ]
    return keras.Sequential(model)


def main():
    graph = generator.generateGraph("WS", 20, 4, 0.5)
    model = createNetwork(graph, [5, 5, 3])
    model.build()
    model.summary()


if __name__ == '__main__':
    main()