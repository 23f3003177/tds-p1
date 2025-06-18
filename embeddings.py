from transformers import AutoModel

# Initialize the model
model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)


def generate_embeddings(inputs, input_type):
    truncate_dim = None
    if input_type == "text":
        return model.encode_text(inputs, truncate_dim=truncate_dim)
    elif input_type == "image":
        model.encode_image(
            inputs, truncate_dim=truncate_dim
        )  # also accepts PIL.Image.Image, local filenames, dataURI
    return None


if __name__ == "__main__":
    vals = generate_embeddings(["Hello"], "text")
# # !pip install transformers onnxruntime pillow
# import onnxruntime as ort
# from transformers import AutoImageProcessor, AutoTokenizer
# import numpy as np
#
# # Load tokenizer and image processor using transformers
# tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)
# image_processor = AutoImageProcessor.from_pretrained(
#     'jinaai/jina-clip-v2', trust_remote_code=True
# )
#
# # Corpus
# sentences = [
#     'غروب جميل على الشاطئ', # Arabic
#     '海滩上美丽的日落', # Chinese
#     'Un beau coucher de soleil sur la plage', # French
#     'Ein wunderschöner Sonnenuntergang am Strand', # German
#     'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
#     'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
#     'Un bellissimo tramonto sulla spiaggia', # Italian
#     '浜辺に沈む美しい夕日', # Japanese
#     '해변 위로 아름다운 일몰', # Korean
# ]
#
# # Public image URLs or PIL Images
# image_urls = ['https://i.ibb.co/nQNGqL0/beach1.jpg', 'https://i.ibb.co/r5w8hG8/beach2.jpg']
#
# # Tokenize input texts and transform input images
# input_ids = tokenizer(sentences, return_tensors='np')['input_ids']
# pixel_values = np.zeros((len(sentences), 3, 512, 512), dtype=np.float32)
#
# # Start an ONNX Runtime Session
# session = ort.InferenceSession('/home/arun/Downloads/model_fp16.onnx')
#
# # Run inference
# output = session.run(None, {'input_ids': input_ids, 'pixel_values': pixel_values })
#
# # Keep the normalised embeddings, first 2 outputs are un-normalized
# _, _, text_embeddings, image_embeddings = output
