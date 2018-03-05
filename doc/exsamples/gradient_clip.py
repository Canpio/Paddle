#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle.v2 as paddle
import paddle.fluid as fluid

# This is an example of how to use gradient clip in MNIST


def the_norml_mnist(image, label):
    """
    This is a normal MNIST without any gradint clip
    """

    hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
    hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')

    predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def clip_a_single_param(image, label):
    """
    The easiest way of using gradient clip is by ParamAttr.
    """

    hidden1 = fluid.layers.fc(input=image, size=128, act='relu')

    # Enables gradint clip for a layer parameter by setting ParamAttr:
    hidden2 = fluid.layers.fc(input=hidden1,
                              size=64,
                              act='relu',
                              param_attr=fluid.ParamAttr(
                                  gradient_clip=fluid.clip.GradientClipByValue(
                                      max=0.01, min=-0.05)))

    predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def clip_a_single_param_ex(image, label):
    """
    You can also use the helper function to enable gradient clips.
    """

    hidden1 = fluid.layers.fc(input=image, size=128, act='relu')

    # Before using the helper function, you need to assign a name to the layer parameter:
    hidden2 = fluid.layers.fc(
        input=hidden1,
        size=64,
        act='relu',
        param_attr=fluid.ParamAttr(name="param_to_be_clip"))

    # Then give the name list of parameters to be clip to the helper function:
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByValue(
            max=0.01, min=-0.05),
        param_list=["param_to_be_clip"])
    # The param_list takes a list, which means it can set gradient clip for more than one paramters in a single time.

    predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def clip_by_group(image, label):
    """
    Sometimes you need to clip gradints by groups. Current, Fluid can support clip by global norm. That is to say, given a group of parameters, their gradients' global norm would not be larger than the given value.
    """

    # Assum that I hope the global norm of hidden1 and hidden2's layer parameters' gradients will be no more than 1.0
    hidden1 = fluid.layers.fc(input=image,
                              size=128,
                              act='relu',
                              param_attr=fluid.ParamAttr(name="param_1"))
    hidden2 = fluid.layers.fc(input=hidden1,
                              size=64,
                              act='relu',
                              param_attr=fluid.ParamAttr(name="param_2"))
    # Uses the helper function to enable global norm gradient clip on param_1 and param_2
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0),
        param_list=["param_1", "param_2"])
    # If 'param_list' is left default(None), all parameters in current program will be include.

    predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


image = fluid.layers.data(name='x', shape=[784], dtype='float32')
label = fluid.layers.data(name='y', shape=[1], dtype='int64')

avg_cost = clip_by_group(image, label)

optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

BATCH_SIZE = 100
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
exe.run(fluid.default_startup_program())

for data in train_reader():
    loss = exe.run(fluid.default_main_program(),
                   feed=feeder.feed(data),
                   fetch_list=[avg_cost])
    print("loss=", loss)
