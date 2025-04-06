import sys
sys.path.append("C:/Users/iceri/OneDrive/Documents/GitHub/Fusion360GalleryDataset/tools/fusion360gym/client")

from fusion360gym_client import Fusion360GymClient
client = Fusion360GymClient("http://127.0.0.1:8080") 

response = client.reconstruct("C:/Users/iceri/OneDrive/Documents/GitHub/Elexus-AddOn/DatasetCreation/ReconstructionDataset/Results/modified_design.json")

response_data = response.json()
print(f"Status: {response_data['status']}")
print(f"Message: {response_data['message']}")
print(f"Data: {response_data.get('data', 'No additional data.')}")