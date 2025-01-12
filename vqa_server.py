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
            
            tokenizer = torch.tensor(huggingface_tokenize_and_pad(hf_tokenizer, question, 32))
            tokenizer = tokenizer.to("cuda")
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                output = model((torch.unsqueeze(image_tensor, dim=0), 
                              torch.unsqueeze(tokenizer, dim=0)))
                
                # Process the output using softmax
                result = process_model_output(output, selected_answers)
                
                print(f"Prediction: {result}")
                return jsonify(result)
                
        except Exception as e:
            return jsonify({'error': str(e)})
    
    app.run(host='0.0.0.0', port=server_port)

if __name__ == "__main__":
    typer.run(main)