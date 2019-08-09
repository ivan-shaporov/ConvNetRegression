import tensorflow as tf

def assert_equal(a, b): tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assert_equal(a, b))

class Learner:
    def __init__(self, batch_size):

        self.n_input = 4
        self.n_output = 1

        self.mean = 0.0
        self.std = 800.0 #/ 100
   
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.learning_rate = 0.001

        self.batch_size = batch_size
        self.feature_size = 1024

    def encoder(self, x, is_training):
         
        assert_equal((self.batch_size, self.feature_size, self.n_input), tf.shape(x))
        
        with tf.device('/gpu:0'):
            with tf.variable_scope("encoder"):

                x = tf.layers.conv1d(x, filters=32, kernel_size=8, strides=1, padding="same", activation=self.tanhLecun, kernel_initializer=tf.initializers.lecun_normal())
                x = tf.layers.conv1d(x, filters=32, kernel_size=8, strides=4, padding="same", activation=self.tanhLecun, kernel_initializer=tf.initializers.lecun_normal()) # 256
                x = tf.layers.conv1d(x, filters=32, kernel_size=8, strides=1, padding="same", activation=self.tanhLecun, kernel_initializer=tf.initializers.lecun_normal())
                x = tf.layers.conv1d(x, filters=32, kernel_size=8, strides=4, padding="same", activation=self.tanhLecun, kernel_initializer=tf.initializers.lecun_normal()) # 64
                x = tf.layers.conv1d(x, filters=32, kernel_size=4, strides=1, padding="same", activation=self.tanhLecun, kernel_initializer=tf.initializers.lecun_normal())
                x = tf.layers.conv1d(x, filters=32, kernel_size=4, strides=2, padding="same", activation=self.tanhLecun, kernel_initializer=tf.initializers.lecun_normal()) # 32
                x = tf.layers.conv1d(x, filters=32, kernel_size=4, strides=1, padding="same", activation=self.tanhLecun, kernel_initializer=tf.initializers.lecun_normal())
                x = tf.layers.conv1d(x, filters=32, kernel_size=4, strides=2, padding="same", activation=self.tanhLecun, kernel_initializer=tf.initializers.lecun_normal()) # 16
                # x = self.batch_norm(x, is_training, name="bn1")
                x = tf.layers.conv1d(x, filters=16, kernel_size=16, strides=16, padding="valid", activation=self.tanhLecun, kernel_initializer=tf.initializers.lecun_normal())#, bias_initializer=tf.initializers.truncated_normal(1.0, 0.25))#, activity_regularizer=tf.contrib.layers.l1_regularizer(scale=0.9))#) # 1
                x = tf.contrib.layers.fully_connected(x, 1, activation_fn=tf.keras.activations.linear, weights_initializer=tf.constant_initializer(40.0/16), biases_initializer=tf.constant_initializer(80.0))

                assert_equal((self.batch_size, 1, 1), tf.shape(x))
                x = tf.reshape(x, [-1])
                return x

    def batch_norm(self, x, is_training, name=None):
        if name is not None:
            with tf.device('/cpu:0'): tf.summary.histogram('pre_batch_norm_%s' % name, x)
        x = tf.contrib.layers.batch_norm(tf.cast(x, tf.float32), center=True, scale=True, is_training=is_training)
        x = tf.cast(x, tf.float64)
        if name is not None:
            with tf.device('/cpu:0'): tf.summary.histogram('post_batch_norm_%s' % name, x)
        return x

    def make_predictor(self, summariesDir):
        ''' executing the result will recreate the model each time.
        see https://github.com/marcsto/rl/blob/master/src/fast_predict2.py for a fix
         '''

        classifier = tf.estimator.Estimator(model_fn=self.encode, model_dir=summariesDir)

        print('predictor initialized from', summariesDir)

        return lambda x: self.run_predictor(classifier, x)

    def run_predictor(self, classifier, batch):
        input_fn = tf.estimator.inputs.numpy_input_fn(x=batch, shuffle=False)
        state = classifier.predict(input_fn, yield_single_examples=True)
        result = list(state)
        return result

    def model(self, features, labels, mode):
        assert_equal([self.batch_size], tf.shape(labels))

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

            labels = tf.cast(labels, tf.float64)
            assert_equal([self.batch_size], tf.shape(labels))

            x = features["x"]
            _, x = self.normalize(x)

            predicted = self.encoder(x, is_training=mode == tf.estimator.ModeKeys.TRAIN)
            assert_equal([self.batch_size], tf.shape(predicted))

            predictionLoss = tf.losses.mean_squared_error(labels=labels, predictions=predicted)

            _, rms_op = tf.metrics.root_mean_squared_error(labels=labels, predictions=predicted)
            _, m_op = tf.metrics.mean_absolute_error(labels=labels, predictions=predicted)

            predictedVar = tf.nn.moments(predicted, axes=[0])[1]
            labelsVar = tf.nn.moments(labels, axes=[0])[1]

            with tf.device('/cpu:0'):
                tf.summary.histogram('predicted', predicted)
                tf.summary.scalar('predictedVar', predictedVar)
                tf.summary.scalar('labelsVar', labelsVar)
                tf.summary.scalar('varRatio', predictedVar / labelsVar)
                tf.summary.scalar('predictionLoss', predictionLoss)
                tf.summary.scalar('rmse', rms_op)
                tf.summary.scalar('me', m_op)
                tf.summary.histogram('labels', labels)
                tf.summary.histogram('err', tf.maximum(tf.minimum(predicted - labels, 10), -10))

            loss = predictionLoss

            if mode == tf.estimator.ModeKeys.TRAIN:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):                
                    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)

                    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            eval_metric_ops = {
                "eval/mae": tf.metrics.mean_absolute_error(labels=labels, predictions=predicted),
                "eval/rmse": tf.metrics.root_mean_squared_error(labels=labels, predictions=predicted),
            }
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    def encode(self, features, labels, mode):
        if mode != tf.estimator.ModeKeys.PREDICT:
            raise "encoder should be used only in predict mode"

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

            _, x = self.normalize(features)
            assert_equal([self.feature_size, self.n_input], x.shape[1:])

            state = self.encoder(x, is_training=False)

        return tf.estimator.EstimatorSpec(mode=mode, predictions=state)

    def toSteps(self, samples):
        first = tf.slice(samples, [0, 0, 0], [-1, 1, -1])
        d = samples[:, 1:, :] - samples[:, :-1, :]
        return (first, d)

    def normalize(self, chunks):
        with tf.name_scope("normalize"):
            f, steps = self.toSteps(chunks)
            steps = tf.maximum(tf.minimum(steps, 1000), -1000)
            #steps = tf.maximum(tf.minimum(steps, 100), -100)
            steps = (steps - self.mean) / self.std
            return (f, steps)

    @staticmethod
    def tanhLecun(x): return tf.nn.tanh(x * 2.0/3.0) * 1.7159

    def schema(self): return "r%gb%d" % (self.learning_rate, self.batch_size)
