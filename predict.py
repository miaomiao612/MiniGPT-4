# import os
# cache = "/src/weights/"
# os.environ["TORCH_HOME"] = "/src/weights/"
# os.environ["HF_HOME"] = "/src/weights/"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/src/weights/"
# if not os.path.exists(cache):
#     os.makedirs(cache)

# Do not import torch before setting the environment variables

import argparse
from PIL import Image
from minigpt4.conversation.conversation import CONV_VISION, Chat
from minigpt4.common.registry import registry
from minigpt4.common.config import Config
#from cog import BasePredictor, Input, Path


class test():

    def __init__(self):
        # Copied as-is from demo.py
        parser = argparse.ArgumentParser(description="Demo")
        parser.add_argument("--cfg-path", required=True,
                            help="path to configuration file.")
        parser.add_argument("--gpu-id", type=int, default=0,
                            help="specify the gpu to load the model.")
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )
        args = parser.parse_args(
            ["--cfg-path", "eval_configs/minigpt4_eval.yaml"])

        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(
            model_config).to('cuda:{}'.format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(
            vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(self.model, self.vis_processor,
                            device='cuda:{}'.format(args.gpu_id))

        self.prefix = "hello"

    def predict(
        self,
        image="hopper.jpg",
        message = "text info",
        num_beams = 1,
        temperature = 0.75,
        max_new_tokens = 500) -> str:
        """Run a single prediction on the model"""
        chat = self.chat
        raw_image = Image.open(image).convert("RGB")
        chat_state = CONV_VISION.copy()
        img_list = []
        llm_message = chat.upload_img(raw_image, chat_state, img_list)
        chat.ask(message, chat_state)

        llm_message = chat.answer(
            conv=chat_state,
            img_list=img_list,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature
        )[0]

        return llm_message

# t = test()
# t.predict()
