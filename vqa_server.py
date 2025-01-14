from typing import Optional
from flask import Flask, request, jsonify, render_template
import numpy as np
import tifffile
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import typer
from configilm.ConfigILM import _get_hf_model as get_huggingface_model
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.util import huggingface_tokenize_and_pad
from lit4rsvqa import LitVisionEncoder

import torch
from torchvision import transforms
import rasterio
import io
import numpy as np



BEN_MEAN = torch.Tensor(
    [
        429.9430203,
        614.21682446,
        590.23569706,
        2218.94553375,
        950.68368468,
        1792.46290469,
        2075.46795189,
        1594.42694882,
        1009.32729131,
        2266.46036911,
        -12.619993741972035,
        -19.29044597721542,
    ]
)
BEN_STD = torch.Tensor(
    [
        572.41639287,
        582.87945694,
        675.88746967,
        1365.45589904,
        729.89827633,
        1096.01480586,
        1273.45393088,
        1079.19066363,
        818.86747235,
        1356.13789355,
        5.115911777546365,
        5.464428464912864,
    ]
)

def  preprocess_image_vqa(img_tensor):
    # img_array = load_img_from_remote(img)
    #img_tensor = torch.from_numpy(img).float()
    # with rasterio.open(img) as src:
    #     image_data = src.read()
    #     num_bands = image_data.shape[0]
    #     print(f"Number of bands: {num_bands}")
    #     img_tensor = image_data.astype(np.float64)

    print(img_tensor.shape)
    # img_tensor = torch.from_numpy(img_tensor).float().to("cpu")

    if img_tensor.shape[0] == 10 and img_tensor.shape[1] == 120 and img_tensor.shape[2] == 120:
        mean, std = BEN_MEAN[:10], BEN_STD[:10]
        resize_size = (120,120)
    else:
        mean, std = BEN_MEAN[10:], BEN_STD[10:]
        resize_size = (120,120)

    preprocess_transform = transforms.Compose([
        transforms.Resize(resize_size, antialias=True),
        transforms.Normalize(mean, std),
    ])
    contents=  preprocess_transform(img_tensor)
    return contents

    # with rasterio.open(img) as src:
    #     output_meta = src.meta.copy()
    #     output_meta.update(compress='lzw')
    #     output_meta.update(count=contents.shape[0])
    # virtual_file = io.BytesIO()

    # with rasterio.open(virtual_file, 'w', **output_meta) as dst:
    #     for i in range(1, contents.shape[0]+1):
    #         dst.write(contents[i-1], i)
    # virtual_file.seek(0)
    # return virtual_file

def load_tif_images_as_tensor(file_path):
    tif_stack = tifffile.imread(file_path)
    tif_stack_float32 = tif_stack.astype(np.float32)
    torch_tensor = torch.from_numpy(tif_stack_float32)
    return torch_tensor

def process_model_output(logits: torch.Tensor, selected_answers: list) -> dict:
    """
    Process model outputs using softmax to get probabilities and predictions.
    
    Args:
        logits: Raw model output tensor of shape (batch_size, 1000)
        selected_answers: List of possible answers matching the 1000 classes
        
    Returns:
        Dictionary with prediction and confidence
    """
    probs = F.softmax(logits, dim=1)
    
    max_prob, pred_idx = torch.max(probs, dim=1)
    
    pred_idx = pred_idx.item()
    confidence = max_prob.item()
    
    predicted_answer = selected_answers[pred_idx]
    
    return {
        'answer': predicted_answer,
        'confidence': confidence,
        'is_binary': predicted_answer.lower() in ['yes', 'no']
    }

def main(
    server_port: int,
    model_checkpoint_path: str,
    vision_model: str = "mobilevit_s",
    text_model: str = "prajjwal1/bert-tiny",
    seed: int = 42,
    matmul_precision: str = "medium",
):
    torch.set_float32_matmul_precision(matmul_precision)
    pl.seed_everything(seed, workers=True)
    
    model = LitVisionEncoder.load_from_checkpoint(model_checkpoint_path)
    model = model.to("cuda")
    
    print(
        f"Model Stats: Params: {model.get_stats()['params']:15,d}\n"
        f"              Flops: {model.get_stats()['flops']:15,d}"
    )
    
    hf_tokenizer, _ = get_huggingface_model(
        model_name=text_model, load_pretrained_if_available=False
    )
    
    selected_answers = ["no", "yes"] + ["answer_" + str(i) for i in range(998)]  # We only care about the first 2. The others are placeholders.
    
    print("=== Loading finished ===")
    
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            file = request.files['image']
            question = request.form['string']
            print(f"Processing question: {question}")
            
            file_path = '/tmp/uploaded_image.tif'
            file.save(file_path)
            
            image_tensor = load_tif_images_as_tensor(file_path)
            image_tensor = image_tensor.permute(2, 0, 1)
            image_tensor = image_tensor.to("cuda")
            
            # print(image_tensor)
            # print("transform")
            # image_tensor = preprocess_image_vqa(image_tensor)
            # print(image_tensor)
            
            tokenizer = torch.tensor(huggingface_tokenize_and_pad(hf_tokenizer, question, 32))
            tokenizer = tokenizer.to("cuda")
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                output = model((torch.unsqueeze(image_tensor, dim=0), 
                              torch.unsqueeze(tokenizer, dim=0)))
                
                # Process the output using softmax
                # print(output)
                
                result = process_model_output(output, selected_answers)
                
                print(f"Prediction: {result}")
                return jsonify({'prediction': result['answer']})
                
        except Exception as e:
            return jsonify({'error': str(e)})
    
    app.run(host='0.0.0.0', port=server_port)

if __name__ == "__main__":
    typer.run(main)