import bentoml
import argparse
import torch

import numpy as np
# import FOTSModel
from bentoml.io import JSON
from bentoml._internal.types import JSONSerializable
from model.kakao import katalk_parsing

from model.emotion import service as emotion
from util.emotion import Emotion
from kobert.pytorch_kobert import get_pytorch_kobert_model
from model.chatbot.kobert import chatbot as emotion_n
from model.emotion import emotion_p
from model.emotion import emotion_pn
from model.emotion import classifier
from model.chatbot.kobert.classifier import KoBERTforSequenceClassfication, kobert_input
import os
# from data_helpers.data_utils import resize_image
# from utils import TranscriptEncoder, classes
import gluonnlp as nlp
from kobert.pytorch_kobert import get_pytorch_kobert_model

kobert_model, vocab = get_pytorch_kobert_model()


from kobert_transformers import get_tokenizer
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _load_emotion(model_path,num_classes):
    """Load model from given path to available device."""
    kobert_model, vocab = get_pytorch_kobert_model()
    model = classifier.BERTClassifier(kobert_model, dr_rate=0.5, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    # model.to(DEVICE)
    return model

# def _load_emotion_pn_model(model_path):
#     """Load model from given path to available device."""
#     kobert_model, vocab = get_pytorch_kobert_model()
#     model = classifier.BERTClassifier(kobert_model, dr_rate=0.5, num_classes=3)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()
#     # tokenizer = get_tokenizer()
#     # model.to(DEVICE)
#     # model.load_state_dict(torch.load(model_path, map_location=DEVICE)["model"])
#     return model
#
def _load_emotion_n_model(model_path):
    """Load model from given path to available device."""
    model = KoBERTforSequenceClassfication()
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def inference(args):
    """FOTS Inference on give images."""

    root_path = '.'
    checkpoint_path = f"{root_path}/checkpoint"
    save_ckpt_path = f"{checkpoint_path}/chatbot_kobert.pth"

    model_p = _load_emotion(args.model_p,num_classes=2)
    model_pn = _load_emotion(args.model_pn,num_classes=3)
    model_n = _load_emotion_n_model(save_ckpt_path)

    # model.eval()
    # model.training=False

    model_p_saved_model = bentoml.pytorch.save(
            model = model_p,
            tag = "model_p",
        signatures={"__call__": {"batchable": False, "batchdim": 0}},
    )
    print(f"Model saved: {model_p_saved_model}")

    model_pn_saved_model = bentoml.pytorch.save(
            model = model_pn,
            tag = "model_pn",
        signatures={"__call__": {"batchable": False, "batchdim": 0}},
    )
    print(f"Model saved: {model_pn_saved_model}")

    model_n_saved_model = bentoml.pytorch.save(
            model = model_n,
            tag = "model_n",
        signatures={"__call__": {"batchable": False, "batchdim": 0}},
    )
    print(f"Model saved: {model_n_saved_model}")
    return model_p_saved_model, model_pn_saved_model, model_n_saved_model


def test_runner(saved_model,input_img = "img_513.jpg",with_img=True,output_dir="./data_folder/output_eval"):
    input_series="abcde"
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)
    # data = kobert_input(tokenizer, input_series, device, 512)

    data = [input_series, '0']
    dataset_another = [data]

    max_len = 100

    transform = nlp.data.BERTSentenceTransform(
        tok, max_seq_length=max_len, pad=True, pair=False)
    print("input_series", input_series, type(input_series))
    token_ids, valid_length, segment_ids = transform([str(input_series)])
    model_p_runner = bentoml.pytorch.get("model_p:latest").to_runner()
    model_p_runner.init_local()


    return
    # input_orig=cv2.imread(input_img)
    # runner = bentoml.pytorch.get(saved_model.tag).to_runner()
    #
    # input_np = cv2.cvtColor(input_orig, cv2.COLOR_BGR2RGB).astype(np.float32)
    # input_np, _, _ = resize_image(input_np, 512)
    # img_arr = np.array(input_np) / 255.0
    # input_arr = np.expand_dims(img_arr, 0).astype("float32")
    # input_arr = np.transpose(input_arr, (0, 3, 1, 2))
    # runner.init_local()
    # score, geometry, preds, boxes, mapping, indices = runner.run(input_arr)
    # return get_transcript(input_img, input_orig, img_arr, preds, boxes, mapping, indices, with_img, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--model_p", default="./checkpoint/emotion_p.pth", type=str,
        help='Path to trained model'
    )
    parser.add_argument(
        "-pn", "--model_pn", default="./checkpoint/emotion_pn.pth", type=str,
        help='Path to trained model'
    )
    # parser.add_argument(
    #     "-m", "--model_n", default="./checkpoint/emotion_n.pth", type=str,
    #     help='Path to trained model'
    # )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="./data_folder/output_eval",
        help="Output directory to save predictions"
    )
    parser.add_argument(
        "-i", "--input_dir", type=str, default="./data_folder/image",
        help="Input directory having images to be predicted"
    )
    args = parser.parse_args()
    model_p_saved_model, model_pn_saved_model, model_n_saved_model = inference(args)
    test_runner(model_p_saved_model)
    # polys, pred_transcripts = test_runner(saved_model,args.input_img,args.with_img,args.output_dir)
    # print("pred_transcripts",pred_transcripts)

