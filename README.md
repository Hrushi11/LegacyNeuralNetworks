# LegacyNeuralNetworks

```
!pip install LegacyNeuralNetworks==0.0.1
```

```
from LegacyNeuralNetworks.Fill import Writer
from LegacyNeuralNetworks import ARTNeuralNetwork

write = Writer('output.txt')
print(write.questions) 
write.getCode('descision_region_perceptron')
```

Choose from: `['activation_function', 'mcculloh_pitt', 'ascii_perceptron', 'descision_region_perceptron', 'recognize_5x3_matrix', 'ann_forward_backward', 'xor_backprop', 'art_network', 'hopfield_network', 'cnn_object_detection', 'cnn_image_classification', 'cnn_tf_implementation', 'mnist_detection']`
