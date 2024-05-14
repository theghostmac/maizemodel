import os

rel_path = "/Users/macbobbychibuzor/workspace/internship/maizemodel/datasets/maize-leaf-disease-dataset/data"
for i in os.listdir(rel_path):
    print(f"{i} {len(os.listdir(f'{rel_path}/{i}'))}")
    
