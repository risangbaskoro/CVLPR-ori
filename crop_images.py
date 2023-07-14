#      CVLPR
#
#      Copyright (c) 2023. Risang Baskoro
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see https://www.gnu.org/licenses/.

import os
import xml.etree.ElementTree as et

import cv2


def get_bounding_boxes(xml_path):
    tree = et.parse(xml_path)
    root = tree.getroot()

    bounding_boxes = []
    for obj in root.findall('object'):
        text = obj.find('name').text.split('-')[-1].upper()
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        bounding_boxes.append((text, (xmin, ymin, xmax, ymax)))

    return bounding_boxes


def crop_images():
    base_path = os.path.expanduser("~/Datasets/cvlp")
    imgs_path = os.path.join(base_path, "images")
    xmls_path = os.path.join(base_path, "annotations")

    # Make a directory called train, skip if it already exists
    os.makedirs("train", exist_ok=True)

    img_names = os.listdir(imgs_path)

    num_images_without_label = 0

    for img_name in img_names:
        xml_path = os.path.join(xmls_path, img_name.split(".")[0] + ".xml")

        if not os.path.exists(xml_path):
            num_images_without_label += 1
        else:
            bounding_boxes = get_bounding_boxes(xml_path)

            img = cv2.imread(os.path.join(imgs_path, img_name))

            for bbox in bounding_boxes:
                name, (x_min, y_min, x_max, y_max) = bbox
                # Crop the portion of the image inside the bounding box
                cropped_image = img[y_min:y_max, x_min:x_max]
                cropped_image = cv2.resize(cropped_image, (136, 36))
                # Save the cropped image
                i = 0
                while os.path.exists(os.path.join(
                        "train",
                        name + "_" + str(i) + '.jpg')):
                    i += 1

                if name == "PLATE":
                    continue

                cv2.imwrite(os.path.join(
                    "train",
                    name + "_" + str(i) + '.jpg'),
                    cropped_image
                )
                print("Saved image: ", name + "_" + str(i) + '.jpg')

    if num_images_without_label > 0:
        print("WARNING: {} images do not have labels".format(num_images_without_label))


if __name__ == "__main__":
    crop_images()
