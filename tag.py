import argparse
import os
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
import huggingface_hub
from pathlib import Path
from tqdm import tqdm

#model list
models = {"swinv3":"SmilingWolf/wd-swinv2-tagger-v3",
          "vitv3":"SmilingWolf/wd-vit-tagger-v3",
          "vitv3-large":"SmilingWolf/wd-vit-large-tagger-v3",
          "convnextv3":"SmilingWolf/wd-convnext-tagger-v3"}


MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

kaomojis = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
]

def load_labels(dataframe):
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def mcut_threshold(probs):
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

class Predictor:
    def __init__(self):
        self.model_target_size = None
        self.last_loaded_repo = None

    def download_model(self, model_repo):
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            MODEL_FILENAME,
        )
        return csv_path, model_path

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)

        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        model = rt.InferenceSession(model_path)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def prepare_image(self, image_path):
        image = Image.open(image_path).convert("RGBA")
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(self, image_path, model_repo, general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled):
        self.load_model(model_repo)

        image = self.prepare_image(image_path)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = ", ".join(sorted_general_strings).replace("(", "\(").replace(")", "\)")

        return sorted_general_strings, rating, character_res, general_res

def process_image(predictor, image_path, args):
    sorted_general_strings, rating, character_res, general_res = predictor.predict( #general_res isn't used because probability doesn't matter
        image_path,
        args.model,
        args.general_t,
        args.general_mcut,
        args.character_t,
        args.character_mcut
    )
    #Write output
    output_path = image_path.with_suffix('.txt')
    with open(output_path, 'w') as f:
        if (args.general):
            f.write(f"{sorted_general_strings}, ")
        if (args.character):
            if character_res:
                for key in character_res:
                    f.write(f"{key}, ")
        if (args.rating):
            f.write(f"{max(rating, key=rating.get)}, ")


def main():
    parser = argparse.ArgumentParser(description="WaifuDiffusion Tagger CLI")
    parser.add_argument("input_path", type=str, help="Path to the input image or folder")
    parser.add_argument("--model", type=str, default="vitv3", help="Model selection (swinv3, convnextv3, vitv3) (default: vitv3)")
    parser.add_argument("--general", action="store_true", help="Save general tags")
    parser.add_argument("--rating", action="store_true", help="Save rating tag")
    parser.add_argument("--character", action="store_true", help="Save character tags")
    parser.add_argument("--general-t", type=float, default=0.35, help="General tags threshold (default: 0.35)")
    parser.add_argument("--character-t", type=float, default=0.85, help="Character tags threshold (default: 0.85)")
    parser.add_argument("--general-mcut", action="store_true", help="Use MCut threshold for general tags")
    parser.add_argument("--character-mcut", action="store_true", help="Use MCut threshold for character tags")
    args = parser.parse_args()

    if (not args.general and not args.rating and not args.character): #if none chosen assume user wants everything
        args.general = True
        args.rating = True
        args.character = True

    args.model = models.get(args.model)
    if (args.model is None):
        args.model = models.get("vitv3")
        print("Specified model not found, defaulting to vitv3")

    predictor = Predictor()
    
    input_path = Path(args.input_path)
    if input_path.is_file():
        process_image(predictor, input_path, args)
    elif input_path.is_dir():
        image_files = [f for f in input_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        total_images = len(image_files)

        with tqdm(total=total_images, unit="image", desc="Processing images") as pbar:
            for image_path in image_files:
                process_image(predictor, image_path, args)
                pbar.update(1)
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == "__main__":
    main()