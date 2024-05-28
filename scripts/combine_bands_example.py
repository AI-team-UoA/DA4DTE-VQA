import rasterio
import torch
import torch.nn.functional as F
import numpy as np
import tifffile

def load_tif_images_as_tensor(file_path):
    tif_stack = tifffile.imread(file_path)
    tif_stack_float32 = tif_stack.astype(np.float32)
    torch_tensor = torch.from_numpy(tif_stack_float32)
    return torch_tensor

def save_tensor_as_tiff(tensor, output_file):
    # Convert the processed tensor to numpy array
    img_data_np = tensor.numpy()
    
    # Save the tensor as a TIFF file using Rasterio
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=img_data_np.shape[1],
        width=img_data_np.shape[2],
        count=img_data_np.shape[0],
        dtype=img_data_np.dtype
    ) as dst:
        for i in range(img_data_np.shape[0]):
            dst.write(img_data_np[i, :, :], i + 1)
    print(f'Saved processed image as {output_file}')

def check_number_of_bands(file_path):
    # Open the TIFF file
    with rasterio.open(file_path) as dataset:
        # Get the number of bands
        num_bands = dataset.count
        print(f'The image has {num_bands} bands.')

#############
# sentinel-2
#############

# VERY IMPORTANT
# THE FOLLOWING CODE IS ONLY FOR DEMONSTRATING THE FUNCTIONALITY!
# THE CORRECT BAND COMBINATIONS PER THE TRAINING DATA ARE AS FOLLOWS:
# "S2": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]

# Open the TIFF files
images = []
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B01.tif"))
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B02.tif"))
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B03.tif"))
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B04.tif"))
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B05.tif"))
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B06.tif"))
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B07.tif"))
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B08.tif"))
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B8A.tif"))
images.append(load_tif_images_as_tensor("./S2A_MSIL2A_20171101T094131_63_80/S2A_MSIL2A_20171101T094131_63_80_B09.tif"))

# Process the images
img_data = [
    F.interpolate(
        torch.Tensor(np.float32(x)).unsqueeze(dim=0).unsqueeze(dim=0),
        size=(120, 120),  # Ensure you specify height and width as a tuple
        mode="bicubic",
        align_corners=True,
    )
    for x in images
]
img_data = torch.cat(img_data, dim=1).squeeze(dim=0)

# Save the processed tensor as a TIFF file
output_file = 'sentinel2_output_image.tif'
save_tensor_as_tiff(img_data, output_file)

# Check the number of bands in the saved TIFF file
check_number_of_bands(output_file)

#############
# sentinel-1
#############

# VERY IMPORTANT
# THE FOLLOWING CODE IS ONLY FOR DEMONSTRATING THE FUNCTIONALITY!
# THE CORRECT BAND COMBINATIONS PER THE TRAINING DATA ARE AS FOLLOWS:
# "S1": ["VH", "VV"]

# This is how one would prepare sentinel-1 images. Nevertheless, the model only supports sentinel-2 images currently.

# images = []
# images.append(load_tif_images_as_tensor("./S1A_IW_GRDH_1SDV_20180526T045642_34WFU_52_35/S1A_IW_GRDH_1SDV_20180526T045642_34WFU_52_35_VH.tif"))
# images.append(load_tif_images_as_tensor("./S1A_IW_GRDH_1SDV_20180526T045642_34WFU_52_35/S1A_IW_GRDH_1SDV_20180526T045642_34WFU_52_35_VV.tif"))

# # Process the images
# img_data = [
#     F.interpolate(
#         torch.Tensor(np.float32(x)).unsqueeze(dim=0).unsqueeze(dim=0),
#         size=(120, 120),  # Ensure you specify height and width as a tuple
#         mode="bicubic",
#         align_corners=True,
#     )
#     for x in images
# ]
# img_data = torch.cat(img_data, dim=1).squeeze(dim=0)

# # Save the processed tensor as a TIFF file
# output_file = 'sentinel1_output_image.tif'
# save_tensor_as_tiff(img_data, output_file)

# # Check the number of bands in the saved TIFF file
# check_number_of_bands(output_file)

