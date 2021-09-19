from tensorflow.keras import Input, Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dense, LeakyReLU, \
    Conv2DTranspose, Reshape, Flatten
from tensorflow.keras.optimizers import Adam


# Remodelar para se assemelhar com um discriminador de uma DCGAN
def discriminator(input_size=(256, 256, 3)):
    model = Sequential(name="Discriminator")
    # model.add(encoder(input_size))

    model.add(Conv2D(64, 3, activation='relu', padding='same', input_shape=input_size, name="conv_2d_1"))
    model.add(Conv2D(64, 3, activation='relu', padding='same', name="conv_2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_1"))

    model.add(Conv2D(128, 3, activation='relu', padding='same', name="conv_2d_3"))
    model.add(Conv2D(128, 3, activation='relu', padding='same', name="conv_2d_4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_2"))

    model.add(Conv2D(256, 3, activation='relu', padding='same', name="conv_2d_5"))
    model.add(Conv2D(256, 3, activation='relu', padding='same', name="conv_2d_6"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_3"))

    model.add(Conv2D(512, 3, activation='relu', padding='same', name="conv_2d_7"))
    model.add(Conv2D(512, 3, activation='relu', padding='same', name="conv_2d_8"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_4"))

    model.add(Conv2D(1024, 3, activation='relu', padding='same', name="conv_2d_9"))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model


def encoder(input_size=(256, 256, 3)):
    model = Sequential(name="u_net_decoder")

    model.add(Input(input_size))

    model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_1"))
    model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_1"))

    model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_3"))
    model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_2"))

    model.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_5"))
    model.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_6"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_3"))

    model.add(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_7"))
    model.add(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_8"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_4"))

    model.add(Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_9"))
    return model


def decoder(encoder_model):
    Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_8")
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        encoder_model.output)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([encoder_model.get_layer("conv_2d_6"), up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([encoder_model.get_layer("conv_2d_4"), up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([encoder_model.get_layer("conv_2d_2"), up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    model = Model(inputs=[encoder_model.output_shape], outputs=[conv9])
    return model


def u_net(encoder_model, decoder_model, pretrained_weights):
    model = Sequential(name="u_net")
    model.add(encoder_model)
    model.add(decoder_model)
    model.add(Conv2D(1, 1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        encoder_model.load_weights(filepath=pretrained_weights)
    return model


def generator(input_dim=100):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 4 * 4 * 512
    model.add(Dense(n_nodes, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 512)))
    model.add(Conv2DTranspose(256, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, 3, activation='tanh', padding='same'))


def tl_gan(disc, gen):
    disc.trainable = False
    model = Sequential(name="gan")
    model.add(gen)
    model.add(disc)
    opt = Adam(learning_rate=3e-4, beta_1=0.05)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
