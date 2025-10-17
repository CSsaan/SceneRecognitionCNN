import time
import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)



class Inference:
    def __init__(self, model: str, categories_file: str) -> None:
        # onnx model session
        providers=[
            ("CUDAExecutionProvider", # 使用GPU推理
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    # "cudnn_conv_use_max_workspace": "1"  # 在初始化阶段需要占用好几G的显存
                }
            ),
            "CPUExecutionProvider" # 使用CPU推理
        ]
        self.session = ort.InferenceSession(model, providers=providers)
        print("ALL providers:", ort.get_available_providers())
        print("Now Session available providers:", self.session.get_providers())

        self.labels = self.load_categories(categories_file)
        
        # preprocess parameters
        model_inputs = self.session.get_inputs()[0]
        self.input_name = model_inputs.name
        self.img_size = model_inputs.shape[-2:]
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

    def normalize_image(self, image: Image.Image, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        image /= 255
        image = (image - mean) / std
        # return image.astype(np.float32)
        image = image[np.newaxis, ...].astype(np.float32)
        return image

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.img_size)
        image = np.array(image, dtype=np.float32).transpose(2, 0, 1)
        image = self.normalize_image(image, self.mean, self.std)
        return image

    def postprocess(self, prob: np.ndarray) -> str:
        id = np.argmax(prob)
        prob = prob[0].squeeze()
        
        # 实现softmax操作
        prob = np.exp(prob - np.max(prob))  # 数值稳定处理
        prob = prob / np.sum(prob)          # softmax计算

        # output the prediction
        idx = prob.argsort()[::-1]
        probs = prob[idx]
        for i in range(0, 5):
            print(f'{probs[i]:.3f} -> {self.labels[idx[i]]}, idx: {idx[i]+1}')

        return self.labels[id]

    def load_categories(self, categories_file: str) -> list:
        if not os.access(categories_file, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url + ' -P ./docs/')
        classes = list()
        with open(categories_file) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)
        return classes

    def test_latency(self, times=100) -> None:
        total = 0
        inputs = np.random.randn(1, 3, *self.img_size).astype(np.float32)
        for _ in range(times):
            start = time.time()
            _ = self.session.run(None, {self.input_name: inputs})
            end = time.time()
            total += (end - start) * 1000
        print(f"Latency: {total // times}ms")

    def predict(self, img_path: str) -> int:
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        pred = self.session.run(None, {self.input_name: image})
        cls_name = self.postprocess(pred)
        return cls_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./export/outputs/resnet18.onnx')
    parser.add_argument('--source', type=str, default=r'D:\CS\MyProjects\resources\Datasets\CUB_200_2011\val\026.Bronzed_Cowbird\Bronzed_Cowbird_0022_796221.jpg')
    parser.add_argument('--categories_file', type=str, default='./docs/categories_places365.txt')
    args = parser.parse_args()

    session = Inference(args.model, args.categories_file)
    cls_name = session.predict(args.source)
    print(f"{args.source} --> {cls_name.capitalize()}")
    print("Starting Latency test(100 runs averaged)...")
    session.test_latency(times=1000)