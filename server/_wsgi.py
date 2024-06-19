import os
import argparse
import json
import logging
import logging.config

logging.config.dictConfig({
  "version": 1,
  "formatters": {
    "standard": {
      "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": os.getenv('LOG_LEVEL'),
      "stream": "ext://sys.stdout",
      "formatter": "standard"
    }
  },
  "root": {
    "level": os.getenv('LOG_LEVEL'),
    "handlers": [
      "console"
    ],
    "propagate": True
  }
})

from auto_x_ml.api import init_app


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return dict()
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto X')
    parser.add_argument(
        '-p', '--port', dest='port', type=int, default=9090,
        help='Server port')
    parser.add_argument(
        '--host', dest='host', type=str, default='0.0.0.0',
        help='Server host')
    parser.add_argument(
        '--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+', type=lambda kv: kv.split('='),
        help='Additional AutoXMLBase model initialization kwargs')
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='Switch debug mode')
    parser.add_argument(
        '--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None,
        help='Logging level')
    
    parser.add_argument(
        '--groundingdino_model', dest='groundingdino_model', default="../../model_pool/groundingdino_swint_ogc.pth",
        help='Directory where models are stored (relative to the project directory)')
    parser.add_argument(
        '--pose_model', dest='pose_model', default="../../model_pool/unipose_swint.pth",
        help='Directory where models are stored (relative to the project directory)')
    
    parser.add_argument(
        '--check', dest='check', action='store_true',
        help='Validate model instance before launching server')
    parser.add_argument('--basic-auth-user',
                        default=os.environ.get('ML_SERVER_BASIC_AUTH_USER', None),
                        help='Basic auth user')
    
    parser.add_argument('--basic-auth-pass',
                        default=os.environ.get('ML_SERVER_BASIC_AUTH_PASS', None),
                        help='Basic auth pass')    
    
    args = parser.parse_args()

    # setup logging level
    if args.log_level:
        logging.root.setLevel(args.log_level)

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def parse_kwargs():
        param = dict()
        for k, v in args.kwargs:
            if v.isdigit():
                param[k] = int(v)
            elif v == 'True' or v == 'true':
                param[k] = True
            elif v == 'False' or v == 'false':
                param[k] = False
            elif isfloat(v):
                param[k] = float(v)
            else:
                param[k] = v
        return param

    kwargs = get_kwargs_from_config()

    if args.kwargs:
        kwargs.update(parse_kwargs())

    os.environ['groundingdino_model'] = args.groundingdino_model
    os.environ['pose_model'] = args.pose_model

    os.environ['ram_model'] = "../../model_pool/ram_plus_swin_large_14m.pth"
    os.environ['ram_img_size'] = 384

    os.environ['videollama2_model_dir'] = "../../model_pool/VideoLLaMA2-7B-Base"

    os.environ['det_model_dir'] = "../../model_pool/ch_PP-OCRv4_det_server_infer"
    os.environ['cls_model_dir'] = "../../model_pool/ch_ppocr_mobile_v2.0_cls_infer"
    os.environ['rec_model_dir'] = "../../model_pool/ch_PP-OCRv4_rec_server_infer"
    os.environ['table_model_dir'] = "../../model_pool/ch_ppstructure_mobile_v2.0_SLANet_infer"
    os.environ['layout_model_dir'] = "../../model_pool/picodet_lcnet_x1_0_fgd_layout_infer"

    os.environ['layout_dict_path'] = "./auto_x_ml/modules/utils/dict/layout_dict/layout_publaynet_dict.txt"
    os.environ['rec_char_dict_path'] = "./auto_x_ml/modules/utils/ppocr_keys_v1.txt"
    os.environ['table_char_dict_path'] = "./auto_x_ml/modules/utils/dict/table_structure_dict_ch.txt"

    from auto_x import AutoSolution

    if args.check:
        print('Check "' + AutoSolution.__name__ + '" instance creation..')
        model = AutoSolution(**kwargs)

    app = init_app(model_class=AutoSolution, basic_auth_user=args.basic_auth_user, basic_auth_pass=args.basic_auth_pass)

    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    from auto_x import AutoSolution
    # for uWSGI use
    app = init_app(model_class=AutoSolution)
