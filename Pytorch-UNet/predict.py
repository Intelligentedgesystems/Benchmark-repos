import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

Gst.init(None)

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        print(output.shape)
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input-pipeline', '-i', type=str, required=True, help='GStreamer input pipeline')
    parser.add_argument('--output-pipeline', '-o', type=str, required=True, help='GStreamer output pipeline')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


class GstPipeline:
    def __init__(self, pipeline_str, callback):
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.callback = callback

        self.appsink = self.pipeline.get_by_name('appsink')
        self.appsink.connect('new-sample', self.on_new_sample)

        self.appsrc = self.pipeline.get_by_name('appsrc')
        # self.appsrc.set_property('caps', Gst.Caps.from_string('video/x-raw, format=RGB'))
        # self.appsrc.set_property('is-live', True)

        self.loop = GLib.MainLoop()

    def on_new_sample(self, appsink):
        sample = appsink.emit('pull-sample')
        buf = sample.get_buffer()
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if result:
            img = Image.frombytes('RGB', (mapinfo.size[0], mapinfo.size[1]), mapinfo.data, 'raw', 'RGB', 0, 1)
            mask = self.callback(img)
            print(mask)
            if mask is not None:
                mask_buf = Gst.Buffer.new_wrapped(mask.tobytes())
                self.appsrc.emit('push-buffer', mask_buf)
            buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except:
            pass
        self.pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    def process_frame(img):
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        if not args.no_save:
            result = mask_to_image(mask, mask_values)
            return np.array(result)
        return None

    input_pipeline = f'{args.input_pipeline} ! appsink name=appsink'
    output_pipeline = f'appsrc name=appsrc ! {args.output_pipeline}'
    pipeline = GstPipeline(input_pipeline , process_frame)
    pipeline.run()
