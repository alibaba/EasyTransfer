# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
if sys.version_info.major == 2:
  import Queue as queue
else:
  import queue
import threading
import traceback

import abc
import six
import tensorflow as tf
import time

WAIT_TIME = 0.1
POISON_PILL = 'poison_pill'
ALL_PROCESS_EXIT = False

def get_queue(queue_size):
  return queue.Queue(maxsize=queue_size)

def get_time():
    return time.strftime('%H:%M:%S', time.localtime(time.time()))


class Mythread(threading.Thread):
    def __init__(self, name, *args):
        threading.Thread.__init__(self)
        self.func = args[0]
        self.args = args[1]
        self.name = name

    def get_name(self):
        return self.name

    def run(self):
        tf.logging.info("Starting thread %s", self.name)
        t1 = time.time()
        self.func(*self.args)
        t2 = time.time()
        tf.logging.info("Exiting thread %s thread time: %f" % (self.name, t2 - t1))


def print_time(threadName, delay, counter):
    while counter:
        time.sleep(delay)
        counter -= 1


class MultiThread(object):
    def __init__(self, job_name, thread_num, func, args_list, time_out=0):
        """
        Args:
          job_name  name for this multi-thread job
          thread_Num  total number of thread
          func  function to execute for each thread
          arg_list args passed to function
          time_out thread time out setting
        """
        self.job_name = job_name
        self.thread_num = thread_num
        self.func = func
        self.args_list = args_list
        self.time_out = time_out
        if len(self.args_list) <= 1:
            self.thread_pool = [Mythread(job_name + '_%d' % i, func, args_list[0]) for i in range(thread_num)]
        else:
            self.thread_pool = []
            for i in range(self.thread_num):
                self.thread_pool.append(Mythread(job_name + '_%d' % i, func, args_list[i]))

    def run(self):
        for t in self.thread_pool:
            tf.logging.info('thread %s start' % t.get_name())
            t.start()

    def join(self):
        for t in self.thread_pool:
            if self.time_out > 0:
                t.join(self.time_out)
            else:
                t.join()
            tf.logging.info('thread %s has finished' % t.get_name())

        for t in self.thread_pool:
            if t.isAlive():
                tf.logging.error('thread %s not exit correctly.' % t.get_name())


class Process(six.with_metaclass(abc.ABCMeta)):
    """
    base class for all process, including downloading, decoding, inference
    """

    def __init__(self,
                 job_name,
                 thread_num,
                 input_queue=None,
                 output_queue=None,
                 batch_size=1):
        self.input_queue = input_queue
        self.output_queue = output_queue
        assert self.input_queue is None or isinstance(self.input_queue, queue.Queue), \
            'input queue should be a threading queue, but now is %s' % type(self.input_queue)
        assert self.output_queue is None or isinstance(self.output_queue, queue.Queue), \
            'output queue should be a threading queue, but now is %s' % type(self.output_queue)
        self.job_name = job_name
        self.thread_num = thread_num
        self.batch_size = batch_size

        self.num_finished = 0
        self.lock = threading.Lock()
        self.exit = False
        self.abnormal_exit = False

    @abc.abstractmethod
    def process(self, in_data):
        """
        method need to be reimplemented, one can add result to output_queue in this func or
          just return the result
        Args:
          in_data   if self.batch_size is 1, input_data contained only one sample data with arbitrary python type
                    if self.batch_size greater than 1, input_data is a list of data arbitrary python type
        Return
          if None, result should be added to output_queue in this func
          if not None, the returned result will be add to output_queue automatically

        """
        pass

    def destroy(self):
        """
        destroy resources that has been used for this process
        """
        pass

    def run(self):

        def noinput_thread_func():
            global ALL_PROCESS_EXIT
            while not self.exit and not ALL_PROCESS_EXIT:
                try:
                    out = self.process(None)
                    if self.output_queue is not None and out is not None:
                        self.put(out)
                except IndexError:
                    break
                except Exception:
                    tf.logging.info(
                        'Exception occured in thread\n %s' % traceback.format_exc())
                    self.abnormal_exit = True
                    ALL_PROCESS_EXIT = True
                    break
            self.lock.acquire()
            self.num_finished += 1
            if self.thread_num == self.num_finished or self.abnormal_exit:
                self.exit = True
                self.put(POISON_PILL)
            self.lock.release()

        def thread_func():
            global ALL_PROCESS_EXIT
            while not self.exit and not ALL_PROCESS_EXIT:
                input_list = []
                no_data = False
                try:
                    for i in range(self.batch_size):
                        input_data = self.get()
                        if input_data == POISON_PILL:
                            # POISON_PILL indicates that input_queue is empty now,
                            # so input_queue.put can not be blocked
                            self.input_queue.put(POISON_PILL)
                            no_data = True
                            break
                        input_list.append(input_data)
                    if len(input_list) > 0:
                        if self.batch_size == 1:
                            out = self.process(input_list[0])
                        else:
                            out = self.process(input_list)
                        if self.output_queue is not None and out is not None:
                            self.put(out)
                except Exception:
                    tf.logging.info(
                        'Exception occured in thread\n %s' % traceback.format_exc())
                    self.abnormal_exit = True
                    ALL_PROCESS_EXIT = True
                    break
                if no_data:
                    break

            self.lock.acquire()
            self.num_finished += 1
            if self.thread_num == self.num_finished or self.abnormal_exit:
                self.exit = True
                if self.output_queue is not None:
                    self.put(POISON_PILL)
            self.lock.release()

        func = noinput_thread_func if self.input_queue is None else thread_func
        self.multi_threads = MultiThread(self.job_name, self.thread_num, func, [[]])
        self.multi_threads.run()

    def get(self):
        """get data to input queue"""
        global ALL_PROCESS_EXIT
        while not ALL_PROCESS_EXIT:
            try:
                data = self.input_queue.get(timeout=10)
                return data
            except queue.Empty:
                continue
        # when ALL_PROCESS_EXIT, we return POISON_PILL to notify threads exit
        return POISON_PILL

    def put(self, data):
        """put data to output queue"""
        global ALL_PROCESS_EXIT
        while not ALL_PROCESS_EXIT:
            try:
                self.output_queue.put(data, timeout=10)
                break
            except queue.Full:
                continue

    def join(self):
        self.multi_threads.join()
        self.destroy()
        if self.abnormal_exit:
            raise RuntimeError('Process %s failed' % self.job_name)


class Counter(object):
    def __init__(self, interval=100):
        self.cnt = 0
        self.interval = interval

    def count(self):
        # count function is not thread safe, make sure to use it
        # only in process which has one thread like io process
        self.cnt += 1
        if self.cnt > 0 and self.cnt % self.interval == 0:
            tf.logging.info('%d batches have been processed' % self.cnt)

class ProcessExecutor(object):
    def __init__(self, queue_size):
        """
        executor to manage ev_process running, which include automatically creating input output queue for each
          process, start each process, wait for all processes to be finished
        Args:
          queue_size size of queue used for each process
        """
        self.queue_size = queue_size
        self._process_list = []

    def get_input_queue(self):
        assert len(self._process_list) > 0, 'no process is added to ProcessExecutor'
        return self._process_list[-1].output_queue

    def get_output_queue(self):
        return get_queue(self.queue_size)

    def add(self, process):
        """
        add process to executor
        """
        self._process_list.append(process)

    def run(self):
        for proc in self._process_list:
            proc.run()

    def wait(self):
        for proc in self._process_list:
            proc.join()
