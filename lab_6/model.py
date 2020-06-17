from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization


class Autoencoder:
	# Размерность кодированного представления
	encoding_dim = 49

	def build(self, height, width, channels):
		# Энкодер
		# вход encoder
		input_img = Input(shape=(height, width, channels))

		# Encoder
		e_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
		pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)
		batchnorm_1 = BatchNormalization()(pool1)
		e_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(batchnorm_1)
		pool2 = MaxPooling2D((2, 2), padding='same')(e_conv2)
		batchnorm_2 = BatchNormalization()(pool2)
		e_conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(batchnorm_2)
		h = MaxPooling2D((2, 2), padding='same')(e_conv3)

		# Decoder
		d_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
		up1 = UpSampling2D((2, 2))(d_conv1)
		d_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
		up2 = UpSampling2D((2, 2))(d_conv2)
		d_conv3 = Conv2D(16, (3, 3), activation='relu')(up2)
		up3 = UpSampling2D((2, 2))(d_conv3)
		decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)

		autoencoder = Model(input_img, decoded, name="autoencoder")
		return autoencoder
