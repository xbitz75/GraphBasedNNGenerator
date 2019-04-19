from __future__ import absolute_import, division, print_function
import graph as generator
from tensorflow.python import keras as keras
from tensorflow.python import Variable
from tensorflow.python import ones
from tensorflow import math
from tensorflow.python import enable_eager_execution

enable_eager_execution()


class TripletBlock(keras.Model):

    def __init__(self, filters=112, strides=1):
        super(TripletBlock, self).__init__(name="triplet")
        self._filters = filters
        self._strides = strides
        self.relu = keras.layers.ReLU()
        self.dw_conv = keras.layers.DepthwiseConv2D(2, strides=strides, padding="same")
        self.conv = keras.layers.Conv2D(filters, 1)
        self.batch = keras.layers.BatchNormalization()

    def call(self, x, *args):
        x = self.relu(x)
        x = self.dw_conv(x)
        x = self.conv(x)
        return self.batch(x)


class NodeToLayer(keras.Model):

    def __init__(self, node):
        super(NodeToLayer, self).__init__(name="nodeToNN")
        self.node_type = node.type
        self.inputs = node.inputs
        self.inputs_len = len(self.inputs)
        if self.inputs_len > 1:
            self.we_sum = Variable(initial_value=ones(self.inputs_len))
        if self.node_type == "input_node":
            self.block = TripletBlock(strides=2)
        else:
            self.block = TripletBlock(strides=1)

    def call(self, *inputs):
        if self.inputs_len > 1:
            x = math.sigmoid(self.we_sum[0]) * inputs[0]
            for i in range(1, self.inputs_len):
                y = math.sigmoid(self.we_sum[i]) * inputs[i]
                x = x + y
        else:
            x = inputs[0]
        x = self.block(x)
        return x


class Stage(keras.Model):

    def __init__(self, graph):
        super(Stage, self).__init__(name="stage")
        self.nodes = generator.getNodes(graph)
        self.input_nodes = generator.getInputNodes(graph)
        self.output_nodes = generator.getOutputNodes(graph)
        self.stage = []
        for node in self.nodes:
            self.stage.append(NodeToLayer(node))

    def call(self, x, *args):
        results = {}
        for id in self.input_nodes:
            results[id] = self.stage[id](x)
        for id, node in enumerate(self.nodes):
            if id not in self.input_nodes:
                results[id] = self.stage[id](*[results[i] for i in node.inputs])
        result = results[self.output_nodes[0]]
        for i, id in enumerate(self.output_nodes):
            if i > 0:
                result = result + results[id] # possible solution padding
        result = result / len(self.output_nodes)
        return result


class Network(keras.Model):

    def __init__(self, type, num_nodes, num_classes = 3472):
        super(Network, self).__init__()
        self.start = keras.layers.DepthwiseConv2D(3, strides=2)
        self.batch = keras.layers.BatchNormalization()
        if type == "small":
            self.stage_start = TripletBlock(strides=2)
            graph = generator.generateGraph("WS", num_nodes, 4, 0.5)
            generator.drawGraph(graph)
            self.stage = Stage(graph)
            graph = generator.generateGraph("WS", num_nodes, 4, 0.5)
            generator.drawGraph(graph)
            self.stage2 = Stage(graph)
            graph = generator.generateGraph("WS", num_nodes, 4, 0.5)
            generator.drawGraph(graph)
            self.stage3 = Stage(graph)
            self.relu = keras.layers.ReLU()
            self.conv = keras.layers.Conv2D(109*4, 1)
            self.batch2 = keras.layers.BatchNormalization()
        elif type == "normal":
            graph = generator.generateGraph("WS", num_nodes, 4, 0.5)
            generator.drawGraph(graph)
            self.stage_start = Stage(graph)
            graph = generator.generateGraph("WS", num_nodes, 4, 0.5)
            generator.drawGraph(graph)
            self.stage = Stage(graph)
            graph = generator.generateGraph("WS", num_nodes, 4, 0.5)
            generator.drawGraph(graph)
            self.stage2 = Stage(graph)
            graph = generator.generateGraph("WS", num_nodes, 4, 0.5)
            generator.drawGraph(graph)
            self.stage3 = Stage(graph)
            self.relu = keras.layers.ReLU()
            self.conv = keras.layers.Conv2D(109 * 4, 1)
            self.batch2 = keras.layers.BatchNormalization()
        self.avrg = keras.layers.AveragePooling2D(7, 1)
        self.end = keras.layers.Dense(num_classes)

    def call(self, x, *args):
        self.start(x)
        self.batch(x)
        self.stage_start(x)
        self.stage(x)
        self.stage2(x)
        self.stage3(x)
        self.relu(x)
        self.conv(x)
        self.batch2(x)
        self.avrg(x)
        self.end(x)
        return x


def main():
    model = Network("normal", 20)
    model.build(input_shape=(None, 128, 128, 3))
    model.summary()


if __name__ == '__main__':
    main()