import tensorflow as tf
import time
from easytransfer.utils.hooks import avgloss_logger_hook
import whale as wh
import os


class WhaleEstimator(object):
    def __init__(self, model_fn, model_dir, num_model_replica,
                 num_accumulated_batches, keep_checkpoint_max, save_checkpoints_steps,
                 task_index=0):
        self._build_model_fn = model_fn
        self.model_dir = model_dir
        self.num_model_replica = num_model_replica
        self.num_accumulated_batches = num_accumulated_batches
        self.keep_checkpoint_max = keep_checkpoint_max
        self.save_checkpoints_steps = save_checkpoints_steps
        #assert self.save_checkpoints_steps is not None, "save_checkpoints_steps is not None"
        if self.save_checkpoints_steps is None:
            self.save_checkpoints_steps = 100
        self.task_index = task_index

    def get_session(self, sess):
        session = sess
        while type(session).__name__ != 'Session':
            # pylint: disable=W0212
            session = session._sess
        return session

    # DP + GA
    def train(self, input_fn, max_steps):
        #cluster = wh.cluster()
        """
        cluster = wh.cluster(layout={"specific": [[
            ["/job:worker/replica:0/task:0/device:GPU:0"],
            ["/job:worker/replica:0/task:0/device:GPU:1"],
            ["/job:worker/replica:0/task:0/device:GPU:2"],
            ["/job:worker/replica:0/task:0/device:GPU:3"],
            ["/job:worker/replica:0/task:1/device:GPU:0"],
            ["/job:worker/replica:0/task:1/device:GPU:1"],
            ["/job:worker/replica:0/task:1/device:GPU:2"],
            ["/job:worker/replica:0/task:1/device:GPU:3"],
            ["/job:worker/replica:0/task:2/device:GPU:0"],
            ["/job:worker/replica:0/task:2/device:GPU:1"],
            ["/job:worker/replica:0/task:2/device:GPU:2"],
            ["/job:worker/replica:0/task:2/device:GPU:3"],
            ["/job:worker/replica:0/task:3/device:GPU:0"],
            ["/job:worker/replica:0/task:3/device:GPU:1"],
            ["/job:worker/replica:0/task:3/device:GPU:2"],
            ["/job:worker/replica:0/task:3/device:GPU:3"]], [
            ["/job:worker/replica:0/task:0/device:GPU:4"],
            ["/job:worker/replica:0/task:0/device:GPU:5"],
            ["/job:worker/replica:0/task:0/device:GPU:6"],
            ["/job:worker/replica:0/task:0/device:GPU:7"],
            ["/job:worker/replica:0/task:1/device:GPU:4"],
            ["/job:worker/replica:0/task:1/device:GPU:5"],
            ["/job:worker/replica:0/task:1/device:GPU:6"],
            ["/job:worker/replica:0/task:1/device:GPU:7"],
            ["/job:worker/replica:0/task:2/device:GPU:4"],
            ["/job:worker/replica:0/task:2/device:GPU:5"],
            ["/job:worker/replica:0/task:2/device:GPU:6"],
            ["/job:worker/replica:0/task:2/device:GPU:7"],
            ["/job:worker/replica:0/task:3/device:GPU:4"],
            ["/job:worker/replica:0/task:3/device:GPU:5"],
            ["/job:worker/replica:0/task:3/device:GPU:6"],
            ["/job:worker/replica:0/task:3/device:GPU:7"]]]})


        """
        cluster = wh.cluster(layout={"row": 2})
        tf.logging.info('cluster {}'.format(cluster))
        with cluster:
            with wh.replica():
                with wh.pipeline(num_micro_batch=self.num_accumulated_batches):
                    with wh.stage():
                        dataset = input_fn()
                        iterator = dataset.make_initializable_iterator()
                        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
                        results = iterator.get_next()
                        wh.current_scope_as_default()
                    total_loss, train_op = self._build_model_fn(results, None, "train", None)
                    wh.add_to_collection(total_loss, wh.GraphKeys.GLOBAL_MEAN_OBJECTS)
                    tf.summary.scalar('loss', total_loss)


        merged = tf.summary.merge_all()

        if self.task_index == 0:
            summary_writer = tf.summary.FileWriter(os.path.join(self.model_dir, "train_suammary_output"))

        saver = tf.train.Saver(max_to_keep=self.keep_checkpoint_max, var_list=tf.trainable_variables())
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=1024,
            inter_op_parallelism_threads=1024,
            gpu_options=tf.GPUOptions(allow_growth=True,
                                      force_gpu_compatible=True,
                                      per_process_gpu_memory_fraction=1.0))

        avgloss_hook = avgloss_logger_hook(max_steps,
                                           total_loss,
                                           self.model_dir,
                                           100,
                                           self.task_index)

        hooks = [tf.train.StopAtStepHook(last_step=max_steps), avgloss_hook]
        with tf.train.MonitoredTrainingSession(
                hooks=hooks,
                config=session_config) as sess:
            starttime = time.time()
            while not sess.should_stop():
                train_loss, _, step, train_loss_summary = \
                    sess.run([total_loss, train_op, tf.train.get_or_create_global_step(), merged])
                if step % 100 == 0:
                    endtime = time.time()
                    tf.logging.info("loss = {}, step = {} ({} sec)".format(train_loss, step, endtime - starttime))
                    starttime = time.time()

                if step % 100 == 0 and self.task_index == 0:
                    summary_writer.add_summary(train_loss_summary, step)

                if step % self.save_checkpoints_steps == 0 and self.task_index == 0:
                    tf.logging.info("Saving checkpoints for {} into {}".format(step, os.path.join(self.model_dir, 'model.ckpt')))
                    saver.save(self.get_session(sess), os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

        if self.task_index == 0:
            summary_writer.close()

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def export_savedmodel(self):
        raise NotImplementedError
