import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import layers

# Code is based on the following:
# https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/legacy/vit.py

class PatchingAndEmbedding(tf.keras.layers.Layer):
    """
    Layer to patchify images, prepend a class token, positionally embed and
    create a projection of patches for Vision Transformers

    The layer expects a batch of input images and returns batches of patches,
    flattened as a sequence and projected onto `project_dims`. If the height and
    width of the images aren't divisible by the patch size, the supplied padding
    type is used (or 'VALID' by default).

    Reference:
        An Image is Worth 16x16 Words: Transformers for Image Recognition at
        Scale by Alexey Dosovitskiy et al. (https://arxiv.org/abs/2010.11929)

    Args:
        project_dim: the dimensionality of the project_dim
        patch_size: the patch size
        padding: default 'VALID', the padding to apply for patchifying images

    Returns:
        Patchified and linearly projected input images, including a prepended
        learnable class token with shape (batch, num_patches+1, project_dim)

    Basic usage:

    ```
    images = #... batch of images
    encoded_patches = keras_cv.layers.PatchingAndEmbedding(
        project_dim=project_dim,
        patch_size=patch_size)(patches)
    print(encoded_patches.shape) # (1, 197, 1024)
    ```
    """

    def __init__(self, project_dim, patch_size, padding="VALID", **kwargs):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.patch_size = patch_size
        self.padding = padding
        if patch_size < 0:
            raise ValueError(
                "The patch_size cannot be a negative number. Received "
                f"{patch_size}"
            )
        if padding not in ["VALID", "SAME"]:
            raise ValueError(
                f"Padding must be either 'SAME' or 'VALID', but {padding} was "
                "passed."
            )
        self.projection = tf.keras.layers.Conv2D(
            filters=self.project_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding=self.padding,
        )

    def build(self, input_shape):
        self.class_token = self.add_weight(
            shape=[1, 1, self.project_dim], name="class_token", trainable=True
        )
        self.num_patches = (
            input_shape[1]
            // self.patch_size
            * input_shape[2]
            // self.patch_size
        )
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches + 1, output_dim=self.project_dim
        )

    def call(
        self,
        images
    ):
        """Calls the PatchingAndEmbedding layer on a batch of images.
        Args:
            images: A `tf.Tensor` of shape [batch, width, height, depth]

        Returns:
            `A tf.Tensor` of shape [batch, patch_num+1, embedding_dim]
        """
        # Turn images into patches and project them onto `project_dim`
        patches = self.projection(images)
        patch_shapes = tf.shape(patches)
        patches_flattened = tf.reshape(
            patches,
            shape=(
                patch_shapes[0],
                patch_shapes[-2] * patch_shapes[-2],
                patch_shapes[-1],
            ),
        )

        # Add learnable class token before linear projection and positional
        # embedding
        flattened_shapes = tf.shape(patches_flattened)
        class_token_broadcast = tf.cast(
            tf.broadcast_to(
                self.class_token,
                [flattened_shapes[0], 1, flattened_shapes[-1]],
            ),
            dtype=patches_flattened.dtype,
        )
        patches_flattened = tf.concat(
            [class_token_broadcast, patches_flattened], 1
        )
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)

        encoded = patches_flattened + self.position_embedding(positions)
        return encoded


class TransformerEncoder(layers.Layer):
    """
    Transformer encoder block implementation as a Keras Layer.

    Args:
        project_dim: the dimensionality of the projection of the encoder, and
            output of the `MultiHeadAttention`
        mlp_dim: the intermediate dimensionality of the MLP head before
            projecting to `project_dim`
        num_heads: the number of heads for the `MultiHeadAttention` layer
        mlp_dropout: default 0.1, the dropout rate to apply between the layers
            of the MLP head of the encoder
        attention_dropout: default 0.1, the dropout rate to apply in the
            MultiHeadAttention layer
        activation: default 'tf.activations.gelu', the activation function to
            apply in the MLP head - should be a function
        layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization`
            layers

    Basic usage:

    ```
    project_dim = 1024
    mlp_dim = 3072
    num_heads = 4

    encoded_patches = keras_cv.layers.PatchingAndEmbedding(
        project_dim=project_dim,
        patch_size=16)(img_batch)
    trans_encoded = keras_cv.layers.TransformerEncoder(project_dim=project_dim,
        mlp_dim = mlp_dim,
        num_heads=num_heads)(encoded_patches)

    print(trans_encoded.shape) # (1, 197, 1024)
    ```
    """

    def __init__(
        self,
        project_dim,
        num_heads,
        mlp_dim,
        mlp_dropout=0.1,
        attention_dropout=0.1,
        activation=tf.keras.activations.gelu,
        layer_norm_epsilon=1e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.mlp_units = [mlp_dim, project_dim]

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.project_dim // self.num_heads,
            dropout=self.attention_dropout,
        )
        self.dense1 = layers.Dense(self.mlp_units[0])
        self.dense2 = layers.Dense(self.mlp_units[1])

    def call(self, inputs):
        """Calls the Transformer Encoder on an input sequence.
        Args:
            inputs: A `tf.Tensor` of shape [batch, height, width, channels]

        Returns:
            `A tf.Tensor` of shape [batch, patch_num+1, embedding_dim]
        """

        if inputs.shape[-1] != self.project_dim:
            raise ValueError(
                "The input and output dimensionality must be the same, but the "
                f"TransformerEncoder was provided with {inputs.shape[-1]} and "
                f"{self.project_dim}"
            )

        x = self.layer_norm1(inputs)
        x = self.attn(x, x)
        x = layers.Dropout(self.mlp_dropout)(x)
        x = layers.Add()([x, inputs])

        y = self.layer_norm2(x)

        y = self.dense1(y)
        if self.activation == keras.activations.gelu:
            y = self.activation(y, approximate=True)
        else:
            y = self.activation(y)
        y = layers.Dropout(self.mlp_dropout)(y)
        y = self.dense2(y)
        y = layers.Dropout(self.mlp_dropout)(y)

        output = layers.Add()([x, y])

        return output


def create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                          project_dim, mlp_dim, num_heads, mlp_dropout, 
                          attention_dropout, num_classes=10):

    
    activation = tf.keras.activations.gelu

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # This layer rescales [0..1] to [-1..1] since ViTs expect [-1..1]
    x = tf.keras.layers.Rescaling(scale=1.0 / 0.5, offset=-1.0, name="rescaling_2")(x)

    encoded_patches = PatchingAndEmbedding(project_dim, patch_size)(x)
    encoded_patches = tf.keras.layers.Dropout(mlp_dropout)(encoded_patches)

    for _ in range(transformer_layer_num):
        encoded_patches = TransformerEncoder(
            project_dim=project_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            mlp_dropout=mlp_dropout,
            attention_dropout=attention_dropout,
            activation=activation,
        )(encoded_patches)

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    outputs = outputs[:, 0] # KC?
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(outputs)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def vitt8(input_shape, num_classes=10):
    ## TINY
    patch_size = 8
    transformer_layer_num = 12
    project_dim = 192
    mlp_dim = 768
    num_heads = 3
    mlp_dropout = 0.0
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)

def vitb8(input_shape, num_classes=10):
    ## BASE 
    patch_size = 8
    transformer_layer_num = 12
    project_dim = 768
    mlp_dim = 3072
    num_heads = 12
    mlp_dropout = 0.0
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)

def vits8(input_shape, num_classes=10):
    ## SMALL
    patch_size = 8
    transformer_layer_num = 12
    project_dim = 384
    mlp_dim = 1536
    num_heads = 6
    mlp_dropout = 0.0
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)

def vitl8(input_shape, num_classes=10):
    ## LARGE
    patch_size = 8
    transformer_layer_num = 24
    project_dim = 1024
    mlp_dim = 4096
    num_heads = 16
    mlp_dropout = 0.1
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)

def vith8(input_shape, num_classes=10):
    ## HUGE
    patch_size = 8
    transformer_layer_num = 32
    project_dim = 1280
    mlp_dim = 5120
    num_heads = 16 
    mlp_dropout = 0.1
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)

def vitt16(input_shape, num_classes=10):
    ## TINY
    patch_size = 16
    transformer_layer_num = 12
    project_dim = 192
    mlp_dim = 768
    num_heads = 3
    mlp_dropout = 0.0
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)

def vits16(input_shape, num_classes=10):
    ## SMALL
    patch_size = 16
    transformer_layer_num = 12
    project_dim = 384
    mlp_dim = 1536
    num_heads = 6
    mlp_dropout = 0.0
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)

def vitb16(input_shape, num_classes=10):
    ## BASE 
    patch_size = 16
    transformer_layer_num = 12
    project_dim = 768
    mlp_dim = 3072
    num_heads = 12
    mlp_dropout = 0.0
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)

def vitl16(input_shape, num_classes=10):
    ## LARGE
    patch_size = 16
    transformer_layer_num = 24
    project_dim = 1024
    mlp_dim = 4096
    num_heads = 16
    mlp_dropout = 0.1
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)

def vith16(input_shape, num_classes=10):
    ## HUGE
    patch_size = 16
    transformer_layer_num = 32
    project_dim = 1280
    mlp_dim = 5120
    num_heads = 16 
    mlp_dropout = 0.1
    attention_dropout = 0.0

    return create_vit_classifier(input_shape, patch_size, transformer_layer_num, 
                                 project_dim, mlp_dim, num_heads, mlp_dropout, 
                                 attention_dropout, num_classes)


if __name__ == "__main__":
    model = vitb16(input_shape=(32, 32, 3))

    model.summary()

    plot_model_filename = "vitb16_plot.png"
    tf.keras.utils.plot_model(model, to_file=plot_model_filename, show_shapes=True)