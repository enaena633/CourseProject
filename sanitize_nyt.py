import os
import sys
import xml.etree.ElementTree as et

root = "nyt_corpus/data/2000"

def parse_single_article(output_file, month, day, day_file):
    xmlFile = root + "/" + month + "/" + day + "/" + day_file
    fileContents = et.parse(xmlFile)
    body_xml = fileContents.find("./body/body.content/*[@class='full_text']")

    if body_xml is None:
        return

    body_str = et.tostring(body_xml, method="text")
    if b"Bush" in body_str or b"Gore" in body_str:
        body_str = body_str.replace(b"\n", b"")
        body_str = f"{2000}/{month}/{day}\t".encode() + body_str.strip() + b"\n"
        output_file.write(body_str.decode("utf-8"))

if __name__ == '__main__':
    output_file = open("sanitized_nyt.tsv", "w")

    for root, month_folders, month_files in os.walk(root):
        for month in month_folders:
            for subroot, day_folders, day_files in os.walk(root + "/" + month):
                for day in day_folders:
                    for subsubroot, no_folders, day_files in os.walk(root + "/" + month + "/" + day):
                        for day_file in day_files:
                            parse_single_article(output_file, month, day, day_file)

    output_file.close()