import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--data_dir', type=str, default="", help='your_data_dir')
    parser.add_argument('--result_path', type=str, default='./results', help='root result directory')
    # training settings
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='num workers')
    parser.add_argument('--img_size', type=int, default=256, help='img size')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='base learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--print_freq', type=int, default=15, help='print frequency')
    parser.add_argument('--trans', type=str, default="I", help="different pre-process")
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    # checkpoint
    parser.add_argument('--checkpoint_path', type=str, default="", help='your checkpoint_path')
    return parser.parse_args()

def parse_args_pred():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default="./weights/loo_photo_best.onnx", help='.onnx model path')
    parser.add_argument('--img_name', type=str, default="live.jpg", help='image name in images folder')
    return parser.parse_args()

def parse_args_convert():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, default="", help='.pth model weight path')
    parser.add_argument('--onnx_path', type=str, default="", help='.onnx model path')
    parser.add_argument('--opset_ver', type=int, default=12, help='onnx opset version')
    parser.add_argument('--img_size', type=int, default=256, help='img size')
    return parser.parse_args()
