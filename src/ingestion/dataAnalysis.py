from networkx import is_path
import pandas as pd
import json
import zipfile
import os
class DataAnalysis:
    
    def __init__(self):
        self.path = "data/listings1"
        self.zip_path = "data/listings.zip"
        self.json_file_path = 'data/listings1/metadata'

    def analyze(self):
        flattened_data = []
        #Extract the data
        if not os.path.exists('data/shopAssist_Cleaned.csv'):
            extracted_data = self.integrateShopData(self.path)
    
            # Clean the data
            cleaned_data = self.cleanData(extracted_data) 
            # Flatten the data
            flattened_data = [self.flatten_record(rec) for rec in cleaned_data if isinstance(rec, dict)]
            # Convert to DataFrame
            df = pd.DataFrame(flattened_data)
            df.to_csv('data/shopAssist_Ori.csv', index=False)
            # drop the columns 
            df.drop(columns=['item_id','model_number','main_image_id','other_image_id', 'domain_name', 'model_year','spin_id','3dmodel_id','finish_type','pattern'],inplace=True)
            df.fillna('', inplace=True)
            # Save the cleaned data to a CSV file
            df.to_csv('data/shopAssist_Cleaned.csv', index=False)
        else:
            df = pd.read_csv('data/shopAssist_Cleaned.csv')
            
        df['combined'] = df.apply(lambda x: ' '.join(x.astype(str)), axis=1)
        return df['combined']
      

    def integrateShopData(self, shop_data):
        
        if (os.path.exists(shop_data)):
            print(f"Path exists: {shop_data}")
            extracted_data = self.readShopData(self.json_file_path)
            print(f"Extracted data: {extracted_data[0]}")
        else:
            print(f"unzipping the file: {self.zip_path}")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # List all files inside the zip
                zip_ref.extractall('data/')
            extracted_data = self.readShopData(self.json_file_path)
        return extracted_data
    

    def readShopData(self, json_file_path):

        extracted_data=[]
        for filename in os.listdir(json_file_path):
            if filename.endswith('.json'):
                print(f"Reading JSON file: {filename}, file_path: {json_file_path}")
                file_path = os.path.join(json_file_path, filename)
                print(f"File path: {file_path}")
                valid_data=self.load_json_from_folder(file_path)
                extracted_data.extend(valid_data)
        print(f"Extracted data in readShopData: {extracted_data[0]}")
        return extracted_data

    def load_json_from_folder(self, file_name):
        valid_data=[]
        # Load your JSON (string or from a file)
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    valid_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
        return valid_data
    
    def cleanData(self, data):
        cleaned_data = []
        # with open("/content/sample_data/listings_1.json", "r", encoding="utf-8") as f:
        for line in data:
            # data = json.loads(line)
            cleaned = self.remove_language_tag(line)
            cleaned_data.append(cleaned)
        return cleaned_data
    
    def remove_language_tag(self,obj):
    # Recursively remove 'language_tag' keys from nested dicts and lists.
        if isinstance(obj, dict):
            return {k: self.remove_language_tag(v) for k, v in obj.items() if k != "language_tag"}
        elif isinstance(obj, list):
            return [self.remove_language_tag(item) for item in obj]
        else:
            return obj

    def flatten_record(self, record):
        flat = {}
        for key, value in record.items():
            if isinstance(value, list) and all(isinstance(v, dict) and "value" in v for v in value):
                # Join multiple values with comma
                flat[key] = ", ".join(str(v["value"]) for v in value)
            else:
                flat[key] = value
        return flat
