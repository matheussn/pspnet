from models import encoder, decoder

encoder_model = encoder()
encoder_model.summary()
decoder_model = decoder(encoder_model)
decoder_model.summary()
