import tensorflow as tf

#
# ResNet 18, 34, 50, 101, and 152 in TensorFlow 2
#
# He, Kaiming, et al. "Deep residual learning for image recognition." (2016) [1]
#  - https://arxiv.org/abs/1512.03385
#


def resnet_basic_block(inputs, num_filters, strides=1, conv_shortcut=False, name=None):
    # Following ResNet building block from Figure 2 [1].
    # Using the [3x3 , 3x3] x n convention. Section 4.1 [1]

    # Between stacks, the subsampling is performed by convolutions with
    # a stride of 2. Section 4.1 [1]
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
    #   activation. Section 3.4 [1]
    x = tf.keras.layers.BatchNormalization(name=name + "_1_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_2_conv2d",
    )(x)
    # We adopt batch normalization right after each convolution and before
    #   activation. Section 3.4 [1]
    x = tf.keras.layers.BatchNormalization(name=name + "_2_bn")(x)

    #
    # conv_shortcut = False:
    #   The identity shortcuts (Eqn.(1)) can be directly used when the input
    #   and output are of the same dimensions (solid line shortcuts in Fig. 3).
    # conv_shortcut = True:
    #   When the dimensions increase (dotted line shortcuts in Fig. 3), we consider
    #   two options: ... (B) The projection shortcut in Eqn.(2) is used to match
    #   dimensions (done by 1×1 convolutions).
    # Section 3.3 "Residual Network" [1]
    #
    if conv_shortcut:
        # The projection shortcut in Eqn.(2) is used to match dimensions
        # (done by 1×1 convolutions). Section 3.3 "Residual Network" [1]
        y = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=1,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name=name + "_0_conv2d",
        )(inputs)
        y = tf.keras.layers.BatchNormalization(name=name + "_0_bn")(y)
    else:
        # The identity shortcuts (Eqn.(1)) can be directly used when the input
        # and output are of the same dimensions. Section 3.3 "Residual Network" [1]
        y = inputs

    out = tf.keras.layers.add([x, y], name=name + "_add")
    out = tf.keras.layers.Activation("relu", name=name + "_out")(out)

    return out


def resnet_complete_block(
    inputs, num_filters, strides=1, conv_shortcut=False, name=None
):
    # Following ResNet building block from Figure 2 [1].
    # Using the [1x1 , 3x3, 1x1] x n convention. Section 4.1 [1]

    # Between stacks, the subsampling is performed by convolutions with
    # a stride of 2. Section 4.1 [1]
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_1_conv2d",
    )(inputs)
    # We adopt batch normalization right after each convolution and before
    #   activation. Section 3.4 [1]
    x = tf.keras.layers.BatchNormalization(name=name + "_1_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)

    # Between stacks, the subsampling is performed by convolutions with
    # a stride of 2. Section 4.1 [1]
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_2_conv2d",
    )(x)
    # We adopt batch normalization right after each convolution and before
    #   activation. Section 3.4 [1]
    x = tf.keras.layers.BatchNormalization(name=name + "_2_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "_2_relu")(x)

    x = tf.keras.layers.Conv2D(
        4 * num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_3_conv2d",
    )(x)
    # We adopt batch normalization right after each convolution and before
    #   activation. Section 3.4 [1]
    x = tf.keras.layers.BatchNormalization(name=name + "_3_bn")(x)

    #
    # conv_shortcut = False:
    #   The identity shortcuts (Eqn.(1)) can be directly used when the input
    #   and output are of the same dimensions (solid line shortcuts in Fig. 3).
    # conv_shortcut = True:
    #   When the dimensions increase (dotted line shortcuts in Fig. 3), we consider
    #   two options: ... (B) The projection shortcut in Eqn.(2) is used to match
    #   dimensions (done by 1×1 convolutions).
    # Section 3.3 "Residual Network" [1]
    #
    if conv_shortcut:
        # The projection shortcut in Eqn.(2) is used to match dimensions
        # (done by 1×1 convolutions). Section 3.3 "Residual Network" [1]
        y = tf.keras.layers.Conv2D(
            4 * num_filters,
            kernel_size=1,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name=name + "_0_conv2d",
        )(inputs)
        y = tf.keras.layers.BatchNormalization(name=name + "_0_bn")(y)
    else:
        # The identity shortcuts (Eqn.(1)) can be directly used when the input
        # and output are of the same dimensions. Section 3.3 "Residual Network" [1]
        y = inputs

    out = tf.keras.layers.add([x, y], name=name + "_add")
    out = tf.keras.layers.Activation("relu", name=name + "_out")(out)

    return out


def resnet(input_shape, num_blocks=(2, 2, 2, 2), num_classes=10, basic=False):
    #
    # resnet_basic_block: 2x 3x3 convolutions (ResNet18 and ResNet34)
    # resnet_complete_block: 1x 1x1 convolution, 1x 3x3 convolution,
    #   1x 1x1 convolution (ResNet50, ResNet101, and ResNet152)
    #

    if basic:
        resnet_block = resnet_basic_block
    else:
        resnet_block = resnet_complete_block

    inputs = tf.keras.layers.Input(shape=input_shape)

    # The first layer is 7×7 convolutions. Section 4.1 [1]
    name = "conv1"

    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name=name + "_pad")(
        inputs
    )

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_conv2d",
    )(x)

    # We adopt batch normalization right after each convolution and before
    # activation. Section 3.4 [1]
    x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "_relu")(x)

    x = tf.keras.layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name=name + "_pool_pad"
    )(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name=name + "_pool_pool")(x)

    # The numbers of filters are {64, 128, 256, 512} respectively for resnet_basic_block.
    # The numbers of filters are {256, 512, 1024, 2048} respectively for resnet_complete_block.
    # Section 4.1 [1]

    name = "conv2"
    # The first stack uses 64 filters. Section 4.1 [1]
    # First block of the first stack uses identity shortcut (strides=1) since the
    #   input size matches the output size of the first layer is 3×3 convolutions.
    #   Section 3.3 "Residual Network" [1]
    x = resnet_block(
        inputs=x, num_filters=64, strides=1, conv_shortcut=True, name=name + "_block1"
    )
    for blocks in range(num_blocks[0] - 1):
        # All other blocks use identity shortcut (strides=1)
        x = resnet_block(
            inputs=x,
            num_filters=64,
            strides=1,
            conv_shortcut=False,
            name=name + "_block" + str(blocks + 2),
        )

    name = "conv3"
    # The second stack uses 128 filters. Section 4.1 [1]
    # First block of the second stack uses projection shortcut (strides=2)
    #   Section 3.3 "Residual Network" [1]
    x = resnet_block(
        inputs=x, num_filters=128, strides=2, conv_shortcut=True, name=name + "_block1"
    )
    for blocks in range(num_blocks[1] - 1):
        # All other blocks use identity shortcut (strides=1)
        x = resnet_block(
            inputs=x,
            num_filters=128,
            strides=1,
            conv_shortcut=False,
            name=name + "_block" + str(blocks + 2),
        )

    name = "conv4"
    # The third stack uses 256 filters. Section 4.1 [1]
    # First block of the third stack uses projection shortcut (strides=2)
    #   Section 3.3 "Residual N etwork" [1]
    x = resnet_block(
        inputs=x, num_filters=256, strides=2, conv_shortcut=True, name=name + "_block1"
    )
    for blocks in range(num_blocks[2] - 1):
        # All other blocks use identity shortcut (strides=1)
        x = resnet_block(
            inputs=x,
            num_filters=256,
            strides=1,
            conv_shortcut=False,
            name=name + "_block" + str(blocks + 2),
        )

    name = "conv5"
    # The third stack uses 512 filters. Section 4.1 [1]
    # First block of the third stack uses projection shortcut (strides=2)
    #   Section 3.3 "Residual N etwork" [1]
    x = resnet_block(
        inputs=x, num_filters=512, strides=2, conv_shortcut=True, name=name + "_block1"
    )
    for blocks in range(num_blocks[3] - 1):
        # All other blocks use identity shortcut (strides=1)
        x = resnet_block(
            inputs=x,
            num_filters=512,
            strides=1,
            conv_shortcut=False,
            name=name + "_block" + str(blocks + 2),
        )

    # The network ends with a global average pooling, a 10-way fully-connected
    # layer, and softmax. Section 4.1 [1]
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


def resnet18(input_shape, num_classes=10):
    return resnet(input_shape, (2, 2, 2, 2), num_classes=num_classes, basic=True)


def resnet34(input_shape, num_classes=10):
    return resnet(input_shape, (3, 4, 6, 3), num_classes=num_classes, basic=True)


def resnet50(input_shape, num_classes=10):
    return resnet(input_shape, (3, 4, 6, 3), num_classes=num_classes, basic=False)


def resnet101(input_shape, num_classes=10):
    return resnet(input_shape, (3, 4, 23, 3), num_classes=num_classes, basic=False)


def resnet152(input_shape, num_classes=10):
    return resnet(input_shape, (3, 8, 36, 3), num_classes=num_classes, basic=False)


if __name__ == "__main__":
    model = resnet50(input_shape=(224, 224, 3))

    model.summary()

    plot_model_filename = "resnet50_plot.png"
    tf.keras.utils.plot_model(model, to_file=plot_model_filename, show_shapes=True)
