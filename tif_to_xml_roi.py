import cv2
import numpy as np
from lxml import etree

# ==== USER INPUTS ====
input_tif = r"C:\Users\harsh\Downloads\2dDIC_dataset\roi.tif"   # <-- your binary ROI tif file
output_xml = "roi.xml"         # <-- output XML file for DICe

# Load binary mask (ROI tif)
roi_mask = cv2.imread(input_tif, cv2.IMREAD_GRAYSCALE)

# Find contours (white = ROI)
contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Build XML structure
root = etree.Element("ROI")
for contour in contours:
    poly = etree.SubElement(root, "Polygon")
    for pt in contour:
        x, y = pt[0]
        point = etree.SubElement(poly, "Point")
        point.set("x", str(int(x)))
        point.set("y", str(int(y)))

# Save to XML file
tree = etree.ElementTree(root)
tree.write(output_xml, pretty_print=True, xml_declaration=True, encoding="UTF-8")

print(f"✅ ROI XML saved as {output_xml}")
