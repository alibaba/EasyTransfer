import tensorflow as tf
import time
import whale as wh

class WhaleEstimator(object):
    def __init__(self, model_fn, model_dir, num_model_replica, num_accumulated_batches):
        self._build_model_fn = model_fn
        self.model_dir = model_dir
        self.num_model_replica = num_model_replica
        self.num_accumulated_batches = num_accumulated_batches

    def train(self, input_fn, max_steps):
        # row = num_gpus / num_stages
        cluster = wh.cluster(layout={"row": self.num_model_replica})
        """
        cluster = wh.cluster(layout={"specific": [[
            ["/job:worker/replica:0/task:0/device:GPU:0"],
            ["/job:worker/replica:0/task:0/device:GPU:2"],
            ["/job:worker/replica:0/task:0/device:GPU:4"],
            ["/job:worker/replica:0/task:0/device:GPU:6"],
            ["/job:worker/replica:0/task:1/device:GPU:0"],
            ["/job:worker/replica:0/task:1/device:GPU:2"],
            ["/job:worker/replica:0/task:1/device:GPU:4"],
            ["/job:worker/replica:0/task:1/device:GPU:6"],
            ["/job:worker/replica:0/task:2/device:GPU:0"],
            ["/job:worker/replica:0/task:2/device:GPU:2"],
            ["/job:worker/replica:0/task:2/device:GPU:4"],
            ["/job:worker/replica:0/task:2/device:GPU:6"],
            ["/job:worker/replica:0/task:3/device:GPU:0"],
            ["/job:worker/replica:0/task:3/device:GPU:2"],
            ["/job:worker/replica:0/task:3/device:GPU:4"],
            ["/job:worker/replica:0/task:3/device:GPU:6"]], [
            ["/job:worker/replica:0/task:0/device:GPU:1"],
            ["/job:worker/replica:0/task:0/device:GPU:3"],
            ["/job:worker/replica:0/task:0/device:GPU:5"],
            ["/job:worker/replica:0/task:0/device:GPU:7"],
            ["/job:worker/replica:0/task:1/device:GPU:1"],
            ["/job:worker/replica:0/task:1/device:GPU:3"],
            ["/job:worker/replica:0/task:1/device:GPU:5"],
            ["/job:worker/replica:0/task:1/device:GPU:7"],
            ["/job:worker/replica:0/task:2/device:GPU:1"],
            ["/job:worker/replica:0/task:2/device:GPU:3"],
            ["/job:worker/replica:0/task:2/device:GPU:5"],
            ["/job:worker/replica:0/task:2/device:GPU:7"],
            ["/job:worker/replica:0/task:3/device:GPU:1"],
            ["/job:worker/replica:0/task:3/device:GPU:3"],
            ["/job:worker/replica:0/task:3/device:GPU:5"],
            ["/job:worker/replica:0/task:3/device:GPU:7"]]]})
        """

        tf.logging.info('cluster {}'.format(cluster))
        with cluster:
            with wh.replica():
                # global batch size = num_micro_batch * batch_size * model_replica(row))
                # model_replica= total_num_gpu / num_stages
                with wh.pipeline(num_micro_batch=self.num_accumulated_batches):
                    with wh.stage():
                        dataset = input_fn()
                        iterator = dataset.make_initializable_iterator()
                        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
                        results = iterator.get_next()
                        wh.current_scope_as_default()
                    total_loss, train_op = self._build_model_fn(results, None, "train", None)

        with tf.train.MonitoredTrainingSession(
                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            starttime = time.time()
            while not sess.should_stop():
                train_loss, _, step = sess.run([total_loss, train_op, tf.train.get_or_create_global_step()])
                if step % 100 == 0:
                    endtime = time.time()
                    tf.logging.info("loss = {}, step = {} ({} sec)".format(train_loss, step, endtime - starttime))
                    starttime = time.time()

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def export_savedmodel(self):
        raise NotImplementedError
