# Implemented by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com
# 
import os
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



def get_ssdd_dataset_function(data_dir, phase):
    def dataset_function():
        return ssdd_directory_to_detectron_dataset(data_dir, phase)
    return dataset_function


def ssdd_directory_to_detectron_dataset(data_dir, phase):
    image_dir = os.path.join(data_dir, 'JPEGImages')
    label_dir = os.path.join(data_dir, 'Annotations')

    # open imagesets file
    image_set_index_file = os.path.join((os.path.join(data_dir, 'ImageSets/Main')), phase + '.txt')
    assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
    with open(image_set_index_file, 'r') as f:
        lines = f.readlines()
    files = [line.strip() for line in lines]

    images = []
    for i, filename in enumerate(files):
        image_name = os.path.join(image_dir,  filename +'.jpg')
        label_name = os.path.join(label_dir,  filename +'.xml')

        if not ((os.path.isfile(image_name)) and (os.path.isfile(label_name))):
            continue

        annotations = []
        target = ET.parse(label_name).getroot()

        for obj in target.iter('size'):
            imageW = int(obj.find('width').text)
            imageH = int(obj.find('height').text)


        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text)
            robj = obj.find('rotated_bndbox')
            for values in robj:
                mbox_cx = float(robj.find('rotated_bbox_cx').text)  # rbox
                mbox_cy = float(robj.find('rotated_bbox_cy').text)
                mbox_w0 = float(robj.find('rotated_bbox_w').text)
                mbox_h0 = float(robj.find('rotated_bbox_h').text)
                angle = float(robj.find('rotated_bbox_theta').text)


            #conversion from RSDD to Detectron angle format
            #RSDD -90 to +90, zero at West, CW -- see page 584 of RSDD paper
            #SSDD -90 to +90, zero at West, CW -- see page 584 of RSDD paper
            #Detectron : -180 to +180, zero at North, CCW
            #angle = -(90 + angle)
            #angle = (90 - angle)
            if mbox_w0>mbox_h0:
                mbox_w = mbox_h0
                mbox_h = mbox_w0
                angle = angle-90
            else:
                mbox_w = mbox_w0
                mbox_h = mbox_h0

            angle = -angle

            
            annotations.append({
                "bbox_mode": 4,  # Oriented bounding box (cx, cy, w, h, a)
                "category_id": 0,
                "bbox": (mbox_cx, mbox_cy, mbox_w, mbox_h, angle)
            })

        images.append({
                "id": int(i),
                "file_name": image_name,
                "annotations": annotations,
                "width" : imageW,
                "height" : imageH,
                })
    return images
