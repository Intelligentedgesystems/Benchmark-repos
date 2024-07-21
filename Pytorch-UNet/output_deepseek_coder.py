import sys
import gi
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from unet.unet_model import UNet
from utils.data_loading import BasicDataset

gi.require_version('Gst', '1.0')
# gi.
from gi.repository import Gst, GstApp, GObject

Gst.init(None)

class VideoProcessor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.model = UNet(n_channels=3, n_classes=2).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def preprocess(self, frame):
        img = Image.fromarray(frame)
        img = BasicDataset.preprocess(None, img, scale_factor=1.0, is_mask=False)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device, dtype=torch.float32)
        return img
    
    def predict(self, frame):
        img = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(img).cpu()
            output = F.interpolate(output, (frame.shape[0], frame.shape[1]), mode='bilinear')
            mask = output.argmax(dim=1).squeeze().numpy()
            return mask
        
class GstPipeline:
    def __init__(self, processor):
        self.processor = processor
        self.pipeline = Gst.Pipeline.new('pipeline')
        self.appsink = GstApp.AppSink.new()
        self.appsink.set_property('emit-signals', True)
        self.appsink.connect('new-sample', self.on_new_sample)
        
        source = Gst.ElementFactory.make('v4l2src', 'source')
        caps = Gst.Caps.from_string('video/x-raw,format=RGB,width=640,height=480,framerate=30/1')
        filter = Gst.ElementFactory.make('capsfilter', 'filter')
        filter.set_property('caps', caps)
        convert = Gst.ElementFactory.make('videoconvert', 'convert')
        self.pipeline.add(source)
        self.pipeline.add(filter)
        self.pipeline.add(convert)
        self.pipeline.add(self.appsink)
        source.link(filter)
        filter.link(convert)
        convert.link(self.appsink)
        
    def on_new_sample(self, appsink):
        sample = appsink.pull_sample()
        buf = sample.get_buffer()
        caps = sample.get_caps()
        height = caps.get_structure(0).get_value('height')
        width = caps.get_structure(0).get_value('width')
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if result:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3))
            mask = self.processor.predict(frame)
            # Process the mask as needed (e.g., display or save)
            buf.unmap(mapinfo)
            return Gst.FlowReturn.OK
        
        def run(self):
            self.pipeline.set_state(Gst.State.PLAYING)
            try:
                loop = GObject.MainLoop()
                loop.run()
            except:
                pass
            self.pipeline.set_state(Gst.State.NULL)
            
if __name__ == '__main__':
    model_path = 'MODEL.pth'
    processor = VideoProcessor(model_path)
    pipeline = GstPipeline(processor)
    pipeline.run()