import bentoml
from bentoml.io import JSON,Text
from bentoml._internal.types import JSONSerializable
from bentoml.io import NumpyNdarray, Image
from PIL.Image import Image as PILImage
import numpy as np
# from eval_functions import get_transcript
# from data_helpers.data_utils import resize_image
# import cv2
from numpy.typing import NDArray
import typing as t
from model.chatbot.kobert.classifier import kobert_input
import torch
import gluonnlp as nlp
from kobert.pytorch_kobert import get_pytorch_kobert_model

# from bentoml._internal.store import Store
# from bentoml._internal.store import StoreItem

# class DummyStore(Store[DummyItem]):
#     def __init__(self, base_path: "t.Union[PathType, FS]"):
#         super().__init__(base_path, DummyItem)

# store = Store("/root/bentoml/models/fots_model/")
# latest = store.get("fots_model:latest")


from kobert_transformers import get_tokenizer

model_p_runner = bentoml.pytorch.get("model_p:latest").to_runner()
model_pn_runner = bentoml.pytorch.get("model_pn:latest").to_runner()
model_n_runner = bentoml.pytorch.get("model_n:latest").to_runner()

kobert_model, vocab = get_pytorch_kobert_model()

svc = bentoml.Service(name="fots_model_runner", runners=[model_p_runner,model_pn_runner,model_n_runner])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


@svc.api(input=Text(), output=NumpyNdarray(dtype="str"))
async def predict(input_series: str) -> NDArray[t.Any]:
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)
    # data = kobert_input(tokenizer, input_series, device, 512)

    data = [input_series, '0']
    dataset_another = [data]

    max_len=100





    transform = nlp.data.BERTSentenceTransform(
        tok, max_seq_length=max_len, pad=True, pair=False)
    print("input_series",input_series, type(input_series))
    token_ids, valid_length, segment_ids = transform([str(input_series)])




    logit = await model_p_runner.async_run(token_ids, valid_length, segment_ids)
    # polys, pred_transcripts = get_transcript(input_img, input_orig, img_arr, preds, boxes, mapping, indices, False, None)
    # print("pred_transcripts",type(pred_transcripts))
    return np.array(logit)