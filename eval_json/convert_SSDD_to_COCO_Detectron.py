# Written by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com

import os
import sys
from sahi.utils.coco import Coco, CocoCategory, CocoImage
from RotatedCocoAnnotation import CocoAnnotationOBB
from sahi.utils.file import save_json
from PIL import Image

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

#this is to convert SSDD to json format
current_dir = os.path.dirname(os.path.abspath(__file__))

DATA_PATH  = r"/home/mikicil/xo23898/SHIP_DETECTION/DATASET/Official-SSDD-OPEN/RBox_SSDD/voc_style/"
DATASET_phase = 'test_offshore'

image_indexname_file = os.path.join(os.path.join(DATA_PATH, r"ImageSets/Main"), DATASET_phase + '.txt')
image_folder_path = os.path.join(DATA_PATH, 'JPEGImages')
annot_folder_path = os.path.join(DATA_PATH, 'Annotations')

with open(image_indexname_file, 'r') as f:
	lines = f.readlines()
image_names = [line.strip()+'.jpg' for line in lines]


coco = Coco()
#coco.add_category(CocoCategory(id=0, name='Ship'))
coco.add_category(CocoCategory(id=1, name='Ship'))

n_image = 0
n_ships= 0
for image_name in image_names:
	width, height = Image.open(os.path.join(image_folder_path,image_name)).size
	coco_image = CocoImage(file_name=image_name, height=height, width=width)

	xml_objs = ET.parse(os.path.join(annot_folder_path, image_name[:-4] +'.xml')).getroot()



	for obj in xml_objs.iter('object'):
		robj = obj.find('rotated_bndbox')
		for values in robj:
			mbox_cx = float(robj.find('rotated_bbox_cx').text)  # rbox
			mbox_cy = float(robj.find('rotated_bbox_cy').text)
			mbox_w0  = float(robj.find('rotated_bbox_w').text)
			mbox_h0  = float(robj.find('rotated_bbox_h').text)
			angle   = float(robj.find('rotated_bbox_theta').text)

			#conversion from RSDD to Detectron angle format
			#RSDD -90 to +90, zero at West, CW -- see page 584 of RSDD paper
			#RSDD 0 to +90, zero at North, CW
			#Detectron : -180 to +180, zero at North, CCW

			if mbox_w0>mbox_h0:
				mbox_w = mbox_h0
				mbox_h = mbox_w0
				angle = angle-90
			else:
				mbox_w = mbox_w0
				mbox_h = mbox_h0

			angle = -angle

		
		

		coco_image.add_annotation(
		CocoAnnotationOBB(
		#bbox=[mbox_cx, mbox_cy, mbox_w, mbox_h, angle],
		bbox=[mbox_cx, mbox_cy, mbox_w, mbox_h, angle],
		#category_id=0,
		category_id=1,
		category_name='Ship',
			)
		)
		
		n_ships+=1

	coco.add_image(coco_image)
	
	n_image+=1

output_json = os.path.join(current_dir,'SSDD_' + DATASET_phase + '_COCO_OBB_Detectron.json')
save_json(data=coco.json, save_path=output_json)



import json
import time
from collections import OrderedDict

# Load existing JSON file
with open(output_json, "r") as f:
    data = json.load(f)

# Get today's date and year
today_str = time.strftime("%Y-%m-%d")
current_year = int(time.strftime("%Y"))

# Add or overwrite the "info" field
data["info"] = {
    "description": "SSDD COCO",
    "version": "1.0",
    "year": current_year,
    "contributor": "Kamirul",
    "date_created": today_str
}

# Move "info" to
data_ordered = OrderedDict()
data_ordered["info"] = data["info"]
for k, v in data.items():
    if k != "info":
        data_ordered[k] = v

# Save the updated JSON
with open(output_json, "w") as f:
    json.dump(data_ordered, f, indent=2)


print('#'*60)
print('DONE')
print('File list: ',image_indexname_file)
print('Image count\t: ', n_image)
print('Ship count\t: ', n_ships)
print('Saved to\t: ', output_json)
print('#'*60)
