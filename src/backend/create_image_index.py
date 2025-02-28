import os
import sys
src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.append(src_directory)
from pinecone import Pinecone, ServerlessSpec
import time
from transformers import AutoProcessor ,CLIPModel
from backend import request_image
from backend import images_dataset
import torch
from backend import logger
from dotenv import load_dotenv
logger = logger.get_logger()

def create_index():
    load_dotenv()
    api_key=os.environ.get("api_key")
    pc = Pinecone(api_key=api_key)

    index_name = "image-search"
    dimension = 512   
    metric = "cosine"  

    if not pc.has_index(index_name): 
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
            cloud="aws",      
             region="us-east-1"  
            )
        )

        
        while True:
            index = pc.describe_index(index_name)
            if index.status.get("ready",False):
                unsplash_index = pc.Index(index_name)
                return unsplash_index
            else:
                time.sleep(1)
    else:
        unsplash_index=pc.Index(index_name)
        return unsplash_index
    
def add_data_to_database(data_frame):
    unsplash_index = create_index()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for _,data in data_frame.iterrows():
        logger.info("Adding embedding")
        url= data["photo_image_url"]
        img = request_image.get_image_from_url(url)
        url = data["photo_image_url"]
        id = data['photo_id']
        inputs = processor(images=img, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
        embddings = image_features.detach().cpu().numpy().flatten().tolist()

        unsplash_index.upsert(
            vectors=[{
                "id":id,
                "values":embddings,
                "metadata": {
                "url": url,
                "photo_id": id 
            }
           } ],
             namespace="image-search-dataset"
        )
        logger.info("Successfully added image to Pinecone index.")
df = images_dataset.get_df(2500,3000)
add_data_to_database(df)