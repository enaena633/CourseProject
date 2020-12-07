import os
import re
import sys
import xml.etree.ElementTree as et

bush_regex = re.compile('Bush[^a-zA-Z]+')
gore_regex = re.compile('Gore[^a-zA-Z]+')
root = "nyt_corpus/data/2000"

def parse_single_article(month, day, day_file):
    xmlFile = root + "/" + month + "/" + day + "/" + day_file
    fileContents = et.parse(xmlFile)
    body_xml = fileContents.find("./body/body.content/*[@class='full_text']")

    if body_xml is None:
        return ""

    body_paragraphs = body_xml.findall("p")
    filtered_output = ""

    for paragraph in body_paragraphs:
        paragraph_str = et.tostring(paragraph, "unicode", "text")
        if bush_regex.search(paragraph_str) != None or gore_regex.search(paragraph_str) != None:
            paragraph_str = paragraph_str.replace("\n", "").strip()
            filtered_output = filtered_output + paragraph_str + " "
            
    return filtered_output

if __name__ == '__main__':
    output_file = open("consolidated_nyt.tsv", "w")

    for root, month_folders, month_files in os.walk(root):
        for month in month_folders:
            for subroot, day_folders, day_files in os.walk(root + "/" + month):
                for day in day_folders:
                    for subsubroot, no_folders, day_files in os.walk(root + "/" + month + "/" + day):
                        day_document = ""

                        for day_file in day_files:
                            single_document = parse_single_article(month, day, day_file)
                            if single_document != "":
                                day_document = day_document + single_document + " "
                        
                        output_file.write(day_document + "\n")

    output_file.close()