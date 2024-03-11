import tensorflow as tf

# Simple Convolutional Neural Network

def simple_basic_block(inputs, num_filters, strides=1, name=None):
    # Between stacks, the subsampling is performed by convolutions with
    # a stride of 2.
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_1_conv2d",
    )(inputs)
    # We adopt batch normalization right after each convolution and before
    #   activation.
    x = tf.keras.layers.BatchNormalization(name=name + "_1_bn")(x)
    out = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)

    return out


def simple(input_shape, num_blocks=3, num_classes=10):

    inputs = tf.keras.layers.Input(shape=input_shape)

    # The first layer is 3×3 convolutions. 
    name = "conv1"
    x = tf.keras.layers.Conv2D(
        16,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_conv2d",
    )(inputs)
    # We adopt batch normalization right after each convolution and before
    # activation. 
    x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "_relu")(x)

    name = "conv2"
    # The first stack uses 16 filters.
    for blocks in range(num_blocks):
        strides = 1 

        # The first block of each stack has a stride of 2
        if blocks == 0:
            strides = 1
        
        x = simple_basic_block(
            inputs=x,
            num_filters=16,
            strides=strides,
            name=name + "_block" + str(blocks + 2),
        )

    name = "conv3"
    # The second stack uses 32 filters.
    for blocks in range(num_blocks):
        strides = 1 

        # The first block of each stack has a stride of 2
        if blocks == 0:
            strides = 2
        
        x = simple_basic_block(
            inputs=x,
            num_filters=32,
            strides=strides,
            name=name + "_block" + str(blocks + 2),
        )

    name = "conv4"
    # The third stack uses 64 filters. 
    for blocks in range(num_blocks):
        strides = 1 

        # The first block of each stack has a stride of 2
        if blocks == 0:
            strides = 2
        
        x = simple_basic_block(
            inputs=x,
            num_filters=64,
            strides=strides,
            name=name + "_block" + str(blocks + 2),
        )

    # The network ends with a global average pooling, a fully-connected layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def simple4_1(input_shape, num_classes=10):
    return simple(input_shape, 1, num_classes=num_classes)

def simple4_3(input_shape, num_classes=10):
    return simple(input_shape, 3, num_classes=num_classes)

def simple4_5(input_shape, num_classes=10):
    return simple(input_shape, 5, num_classes=num_classes)

def simple4_7(input_shape, num_classes=10):
    return simple(input_shape, 7, num_classes=num_classes)

def simple4_9(input_shape, num_classes=10):
    return simple(input_shape, 9, num_classes=num_classes)

def simple4_11(input_shape, num_classes=10):
    return simple(input_shape, 11, num_classes=num_classes)

def simple4_13(input_shape, num_classes=10):
    return simple(input_shape, 13, num_classes=num_classes)

def simple4_15(input_shape, num_classes=10):
    return simple(input_shape, 15, num_classes=num_classes)

if __name__ == "__main__":
    model = simple4_1(input_shape=(32, 32, 3))

    model.summary()

    plot_model_filename = "simple4_1_plot.png"
    tf.keras.utils.plot_model(model, to_file=plot_model_filename, show_shapes=True)
