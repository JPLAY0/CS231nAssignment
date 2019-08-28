import tensorflow as tf

NUM_CLASSES = 1000

class Fire(tf.keras.Model):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes,name=None):
        super(Fire, self).__init__(name='%s/fire'%name)
        self.inplanes = inplanes
        self.squeeze = tf.keras.layers.Conv2D(squeeze_planes, input_shape=(inplanes,), kernel_size=1, strides=(1,1), padding="VALID", activation='relu',name='squeeze')
        self.expand1x1 = tf.keras.layers.Conv2D(expand1x1_planes, kernel_size=1, padding="VALID", strides=(1,1), activation='relu',name='e11')
        self.expand3x3 = tf.keras.layers.Conv2D(expand3x3_planes, kernel_size=3, padding="SAME", strides=(1,1), activation='relu',name='e33')

    def call(self, x):
        x = self.squeeze(x)
        return tf.concat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], axis=3)


class SqueezeNet(tf.keras.Model):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2,2), padding="VALID", activation='relu', input_shape=(224, 224, 3), name='features/layer0'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name='features/layer2'),
            Fire(64, 16, 64, 64, name='features/layer3'),
            Fire(128, 16, 64, 64, name='features/layer4'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name='features/layer5'),
            Fire(128, 32, 128, 128, name='features/layer6'),
            Fire(256, 32, 128, 128, name='features/layer7'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name='features/layer8'),
            Fire(256, 48, 192, 192, name='features/layer9'),
            Fire(384, 48, 192, 192, name='features/layer10'),
            Fire(384, 64, 256, 256, name='features/layer11'),
            Fire(512, 64, 256, 256, name='features/layer12'),
            tf.keras.layers.Conv2D(self.num_classes, kernel_size=1, padding="VALID",  activation='relu', name='classifier/layer1'),
            tf.keras.layers.AveragePooling2D(pool_size=13, strides=13, padding="VALID", name='classifier/layer3')
            ])

    def call(self, x, save_path=None):
        x = self.net(x)
        scores = tf.reshape(x, (-1, self.num_classes))
        return scores
