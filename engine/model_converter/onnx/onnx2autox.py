#!/usr/bin/env python3
import struct
from enum import IntEnum
from collections import OrderedDict
import numpy as np

import onnx
from onnx import numpy_helper
import onnxruntime

def IsExpand(attribute):
    for attr in attribute:
        if attr.name == "kernel_shape":
            filter_dim = attr.ints
        if attr.name == "strides":
            strides = attr.ints
        if attr.name == "pads":
            paddings = attr.ints
        if attr.name == "dilations":
            dilations = attr.ints
    filter_1 = True
    strides_1 = True
    padding_0 = True
    dilation_1 = True
    for j in range(len(strides)):
        filter_1 = filter_1 and ((int)(filter_dim[j]) == 1)
        strides_1 = strides_1 and (strides[j] == 1)
        padding_0 = padding_0 and (paddings[j] == 0)
        dilation_1 = dilation_1 and (dilations[j] == 1)

    return not (filter_1 and strides_1 and padding_0 and dilation_1)

def get_workspace_size(node, ort_outs):
    for attr in node.attribute:
        if attr.name == "group":
            group = attr.i
        if attr.name == "kernel_shape":
            filter_dim = attr.ints

    in_shape = ort_outs[node.input[0]].shape
    out_shape = ort_outs[node.output[0]].shape
	
    return out_shape[2]*out_shape[3]*filter_dim[0]*filter_dim[1]*in_shape[1]/group

def get_input_shapes(onnx_model):
    input_shapes = {}
    for i in onnx_model.graph.input:
        input_shape = i.type.tensor_type.shape.dim    
        input_shapes[i.name] = [x.dim_value for x in input_shape]

    return input_shapes

def load_onnx(model_file, inputs, img):
    onnx_model = onnx.load(model_file)

    # Check the model
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
        exit(1)
    else:
        print('The model is valid!')

    #get matrix values for the nodes
    initializers = onnx_model.graph.initializer
    onnx_weights = {}
    for initializer in initializers:
        weight = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = weight.flatten()

        # limit the amount of decimal points for more compact examples
        #string_weights = np.empty(weight.shape)
        #for index in np.ndindex(weight.shape):
        #    string_weights[index] = "{:.2f}".format(weight[index])
        #onnx_weights[initializer.name] = string_weights

    nodelist = onnx_model.graph.node

    for node in nodelist:
        if node.op_type == "Constant":
            if node.attribute[0].t.data_type == 1:
                onnx_weights[node.output[0]] = np.frombuffer(node.attribute[0].t.raw_data,
                                        dtype=np.float32)

    for node in onnx_model.graph.node:
        for output in node.output:
            onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    inname = [input.name for input in ort_session.get_inputs()]
    outname = [output.name for output in ort_session.get_outputs()]

    if len(inputs) > 1:
        ort_outs = ort_session.run(outname, {inname[0]: [img],inname[1]: [inputs[1]]})
    else:
        ort_outs = ort_session.run(outname, {inname[0]: [img]})

    ort_outs = OrderedDict(zip(outname, ort_outs))
    for input in ort_session.get_inputs():
        ort_outs[input.name] = input

    nodelist = onnx_model.graph.node

    return nodelist