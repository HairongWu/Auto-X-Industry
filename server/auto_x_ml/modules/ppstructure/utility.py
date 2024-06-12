import os
import sys
import cv2
import numpy as np
import paddle

from paddle import inference

def create_predictor(args, mode):
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == "cls":
        model_dir = args.cls_model_dir
    elif mode == "rec":
        model_dir = args.rec_model_dir
    elif mode == "table":
        model_dir = args.table_model_dir
    elif mode == "layout":
        model_dir = args.layout_model_dir
    else:
        model_dir = args.e2e_model_dir

    if model_dir is None:
        sys.exit(0)
    if args.use_onnx:
        import onnxruntime as ort

        model_file_path = model_dir
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path {}".format(model_file_path))
        if args.use_gpu:
            sess = ort.InferenceSession(
                model_file_path, providers=["CUDAExecutionProvider"]
            )
        else:
            sess = ort.InferenceSession(model_file_path)
        return sess, sess.get_inputs()[0], None, None

    else:
        file_names = ["model", "inference"]
        for file_name in file_names:
            model_file_path = "{}/{}.pdmodel".format(model_dir, file_name)
            params_file_path = "{}/{}.pdiparams".format(model_dir, file_name)
            if os.path.exists(model_file_path) and os.path.exists(params_file_path):
                break
        if not os.path.exists(model_file_path):
            raise ValueError(
                "not find model.pdmodel or inference.pdmodel in {}".format(model_dir)
            )
        if not os.path.exists(params_file_path):
            raise ValueError(
                "not find model.pdiparams or inference.pdiparams in {}".format(
                    model_dir
                )
            )

        config = inference.Config(model_file_path, params_file_path)

        if hasattr(args, "precision"):
            if args.precision == "fp16" and args.use_tensorrt:
                precision = inference.PrecisionType.Half
            elif args.precision == "int8":
                precision = inference.PrecisionType.Int8
            else:
                precision = inference.PrecisionType.Float32
        else:
            precision = inference.PrecisionType.Float32

        if args.use_gpu:
            config.enable_use_gpu(args.gpu_mem, args.gpu_id)

        elif args.use_npu:
            config.enable_custom_device("npu")
        elif args.use_mlu:
            config.enable_custom_device("mlu")
        elif args.use_xpu:
            config.enable_xpu(10 * 1024 * 1024)
        else:
            config.disable_gpu()
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if args.precision == "fp16":
                    config.enable_mkldnn_bfloat16()
                if hasattr(args, "cpu_threads"):
                    config.set_cpu_math_library_num_threads(args.cpu_threads)
                else:
                    # default cpu threads as 10
                    config.set_cpu_math_library_num_threads(10)
        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.delete_pass("matmul_transpose_reshape_fuse_pass")
        if mode == "re":
            config.delete_pass("simplify_with_basic_ops_pass")
        if mode == "table":
            config.delete_pass("fc_fuse_pass")  # not supported for table
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)
        input_names = predictor.get_input_names()
        if mode in ["ser", "re"]:
            input_tensor = []
            for name in input_names:
                input_tensor.append(predictor.get_input_handle(name))
        else:
            for name in input_names:
                input_tensor = predictor.get_input_handle(name)
        output_tensors = get_output_tensors(args, mode, predictor)
        return predictor, input_tensor, output_tensors, config


def get_output_tensors(args, mode, predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    if mode == "rec" and args.rec_algorithm in ["CRNN", "SVTR_LCNet", "SVTR_HGNet"]:
        output_name = "softmax_0.tmp_0"
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors

def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img


def slice_generator(image, horizontal_stride, vertical_stride, maximum_slices=500):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image_h, image_w = image.shape[:2]
    vertical_num_slices = (image_h + vertical_stride - 1) // vertical_stride
    horizontal_num_slices = (image_w + horizontal_stride - 1) // horizontal_stride

    assert (
        vertical_num_slices > 0
    ), f"Invalid number ({vertical_num_slices}) of vertical slices"

    assert (
        horizontal_num_slices > 0
    ), f"Invalid number ({horizontal_num_slices}) of horizontal slices"

    if vertical_num_slices >= maximum_slices:
        recommended_vertical_stride = max(1, image_h // maximum_slices) + 1
        assert (
            False
        ), f"Too computationally expensive with {vertical_num_slices} slices, try a higher vertical stride (recommended minimum: {recommended_vertical_stride})"

    if horizontal_num_slices >= maximum_slices:
        recommended_horizontal_stride = max(1, image_w // maximum_slices) + 1
        assert (
            False
        ), f"Too computationally expensive with {horizontal_num_slices} slices, try a higher horizontal stride (recommended minimum: {recommended_horizontal_stride})"

    for v_slice_idx in range(vertical_num_slices):
        v_start = max(0, (v_slice_idx * vertical_stride))
        v_end = min(((v_slice_idx + 1) * vertical_stride), image_h)
        vertical_slice = image[v_start:v_end, :]
        for h_slice_idx in range(horizontal_num_slices):
            h_start = max(0, (h_slice_idx * horizontal_stride))
            h_end = min(((h_slice_idx + 1) * horizontal_stride), image_w)
            horizontal_slice = vertical_slice[:, h_start:h_end]

            yield (horizontal_slice, v_start, h_start)


def calculate_box_extents(box):
    min_x = box[0][0]
    max_x = box[1][0]
    min_y = box[0][1]
    max_y = box[2][1]
    return min_x, max_x, min_y, max_y


def merge_boxes(box1, box2, x_threshold, y_threshold):
    min_x1, max_x1, min_y1, max_y1 = calculate_box_extents(box1)
    min_x2, max_x2, min_y2, max_y2 = calculate_box_extents(box2)

    if (
        abs(min_y1 - min_y2) <= y_threshold
        and abs(max_y1 - max_y2) <= y_threshold
        and abs(max_x1 - min_x2) <= x_threshold
    ):
        new_xmin = min(min_x1, min_x2)
        new_xmax = max(max_x1, max_x2)
        new_ymin = min(min_y1, min_y2)
        new_ymax = max(max_y1, max_y2)
        return [
            [new_xmin, new_ymin],
            [new_xmax, new_ymin],
            [new_xmax, new_ymax],
            [new_xmin, new_ymax],
        ]
    else:
        return None


def merge_fragmented(boxes, x_threshold=10, y_threshold=10):
    merged_boxes = []
    visited = set()

    for i, box1 in enumerate(boxes):
        if i in visited:
            continue

        merged_box = [point[:] for point in box1]

        for j, box2 in enumerate(boxes[i + 1 :], start=i + 1):
            if j not in visited:
                merged_result = merge_boxes(
                    merged_box, box2, x_threshold=x_threshold, y_threshold=y_threshold
                )
                if merged_result:
                    merged_box = merged_result
                    visited.add(j)

        merged_boxes.append(merged_box)

    if len(merged_boxes) == len(boxes):
        return np.array(merged_boxes)
    else:
        return merge_fragmented(merged_boxes, x_threshold, y_threshold)


def check_gpu(use_gpu):
    if use_gpu and (
        not paddle.is_compiled_with_cuda() or paddle.device.get_device() == "cpu"
    ):
        use_gpu = False
    return use_gpu

def cal_ocr_word_box(rec_str, box, rec_word_info):
    """Calculate the detection frame for each word based on the results of recognition and detection of ocr"""

    col_num, word_list, word_col_list, state_list = rec_word_info
    box = box.tolist()
    bbox_x_start = box[0][0]
    bbox_x_end = box[1][0]
    bbox_y_start = box[0][1]
    bbox_y_end = box[2][1]

    cell_width = (bbox_x_end - bbox_x_start) / col_num

    word_box_list = []
    word_box_content_list = []
    cn_width_list = []
    cn_col_list = []
    for word, word_col, state in zip(word_list, word_col_list, state_list):
        if state == "cn":
            if len(word_col) != 1:
                char_seq_length = (word_col[-1] - word_col[0] + 1) * cell_width
                char_width = char_seq_length / (len(word_col) - 1)
                cn_width_list.append(char_width)
            cn_col_list += word_col
            word_box_content_list += word
        else:
            cell_x_start = bbox_x_start + int(word_col[0] * cell_width)
            cell_x_end = bbox_x_start + int((word_col[-1] + 1) * cell_width)
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)
            word_box_content_list.append("".join(word))
    if len(cn_col_list) != 0:
        if len(cn_width_list) != 0:
            avg_char_width = np.mean(cn_width_list)
        else:
            avg_char_width = (bbox_x_end - bbox_x_start) / len(rec_str)
        for center_idx in cn_col_list:
            center_x = (center_idx + 0.5) * cell_width
            cell_x_start = max(int(center_x - avg_char_width / 2), 0) + bbox_x_start
            cell_x_end = (
                min(int(center_x + avg_char_width / 2), bbox_x_end - bbox_x_start)
                + bbox_x_start
            )
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)

    return word_box_content_list, word_box_list
