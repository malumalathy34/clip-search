import streamlit as st
import torch
import time
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoProcessor, CLIPModel
import requests
from PIL import Image
from io import BytesIO
import os
import sys
src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.append(src_directory)
from backend import images_dataset
import numpy as np

def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def create_index():
    api_key = "pcsk_6DvkVe_FzpkJcXtJ315S4EA17E1vDRUxn8kNA46gnsDAGz7XX77HC6TV164fayHuvV8LTf"  # Replace with your actual API key
    pc = Pinecone(api_key=api_key)
    index_name = "image-search"
    dimension = 512  # Example dimension for CLIP embeddings
    metric = "cosine"
    
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while True:
            index = pc.describe_index(index_name)
            if index.status.get("ready", False):
                return pc.Index(index_name)
            else:
                time.sleep(1)
    else:
        return pc.Index(index_name)

def get_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def add_data_to_database(data_frame):
    unsplash_index = create_index()
    model, processor = get_clip_model()
    
    for _, data in data_frame.iterrows():
        url = data["photo_image_url"]
        img = get_image_from_url(url)
        id = data['photo_id']
        inputs = processor(images=img, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
        embeddings = image_features.detach().cpu().numpy().flatten().tolist()
        
        unsplash_index.upsert(
            vectors=[{
                "id": id,
                "values": embeddings,
                "metadata": {"url": url, "photo_id": id}
            }],
            namespace="image-search-dataset"
        )

def search_by_text(query_text, index):
    model, processor = get_clip_model()
    inputs = processor(text=query_text, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    query_vector = text_features.detach().cpu().numpy().flatten().tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True, namespace="image-search-dataset")
    return results
def search_by_image(image, index):
    model, processor = get_clip_model()
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    query_vector = image_features.detach().cpu().numpy().flatten().tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True, namespace="image-search-dataset")
    return results

def main():
    st.title("Image Search with Pinecone and CLIP")
    index = create_index()
    
    option = st.selectbox("Choose Input Type", ["Text", "Image Upload"])
    
    if option == "Text":
        user_text = st.text_input("Enter your search text")
        if st.button("Search"):
            results = search_by_text(user_text, index)
            for match in results['matches']:
                st.image(match['metadata']['url'], caption=f"Match: {match['metadata']['photo_id']}")
    elif option == "Image Upload":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            if st.button("Search by Image"):
                results = search_by_image(image, index)
                for match in results['matches']:
                   st.image(match['metadata']['url'], caption=f"Match: {match['metadata']['photo_id']}")

    
if __name__ == "__main__":
    main()