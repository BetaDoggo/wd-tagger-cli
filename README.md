# WD-Tagger-CLI
A quick and dirty conversion of the [original space](https://huggingface.co/spaces/SmilingWolf/wd-tagger) for my needs.

# How to run:

```
pip install -r requirements.txt
python tag.py <image-or-directory>
```
If no specific tag categories have been selected it will output general tags, character tags, and ratings. If no model is specified it will default to vitv3. 

# Usage:

```
usage: tag.py [-h] [--model MODEL] [--general] [--rating] [--character] [--general-t GENERAL_T]
              [--character-t CHARACTER_T] [--general-mcut] [--character-mcut]
              input_path

positional arguments:
  input_path            Path to the input image or folder

options:
  -h, --help            show this help message and exit
  --model MODEL         Model selection (swinv3, convnextv3, vitv3, vitv3-large) (default: vitv3)
  --general             Save general tags
  --rating              Save rating tag
  --character           Save character tags
  --general-t GENERAL_T
                        General tags threshold (default: 0.35)
  --character-t CHARACTER_T
                        Character tags threshold (default: 0.85)
  --general-mcut        Use MCut threshold for general tags
  --character-mcut      Use MCut threshold for character tags
```
