import os
from model.chatbot.kogpt2 import chatbot as ch_kogpt2
from model.chatbot.kobert import chatbot as ch_kobert
from model.emotion import service as emotion
from util.emotion import Emotion
from util.depression import Depression
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from kss import split_sentences
from model.kakao import katalk_parsing
import json
app = Flask(__name__)
Emotion = Emotion()
Depression = Depression()

@app.route('/')
def hello():
    return "ML server running"

@app.route('/katalk')
def kakaoTalkService():
    # userText = request.files['file']
    # fileName = userText.filename
    nickname="이현"
    pk = 13
    parsing = katalk_parsing.katalk_msg_parse('./data/KakaoTalkChats.txt', nickname)
    data={
        'PK':pk,
        'katalk_data':json.loads(parsing)
    }
    print(data)
    return data


@app.route('/emotion')
def classifyEmotion():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0 or sentence == '\n':
        return jsonify({
            "emotion_no": 2,
            "emotion": "중립"
        })

    result = emotion.predict(sentence)
    print("[*] 감정 분석 결과: " + Emotion.to_string(result))
    return jsonify({
        "emotion_no": int(result),
        "emotion": Emotion.to_string(result)
    })


@app.route('/diary')
def classifyEmotionDiary():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0 or sentence == '\n':
        return jsonify({
            "joy": 0,
            "hope": 0,
            "neutrality": 0,
            "anger": 0,
            "sadness": 0,
            "anxiety": 0,
            "tiredness": 0,
            "regret": 0,
            "depression": 0
        })

    predict, dep_predict = predictDiary(sentence)
    return jsonify({
        "joy": predict[Emotion.JOY],
        "hope": predict[Emotion.HOPE],
        "neutrality": predict[Emotion.NEUTRALITY],
        "anger": predict[Emotion.ANGER],
        "sadness": predict[Emotion.SADNESS],
        "anxiety": predict[Emotion.ANXIETY],
        "tiredness": predict[Emotion.TIREDNESS],
        "regret": predict[Emotion.REGRET],
        "depression": dep_predict
    })


@app.route('/chatbot/g')
def reactChatbotV1():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0 or sentence == '\n':
        return jsonify({
            "answer": "듣고 있어요. 더 말씀해주세요~ (끄덕끄덕)"
        })

    answer = ch_kogpt2.predict(sentence)
    return jsonify({
        "answer": answer
    })


@app.route('/chatbot/b')
def reactChatbotV2():
    sentence = request.args.get("s")
    if sentence is None or len(sentence) == 0 or sentence == '\n':
        return jsonify({
            "answer": "듣고 있어요. 더 말씀해주세요~ (끄덕끄덕)"
        })

    answer, category, desc, softmax = ch_kobert.chat(sentence)
    return jsonify({
        "answer": answer,
        "category": category,
        "category_info": desc
    })


def predictDiary(s):
    total_cnt = 0.0
    dep_cnt = 0
    predict = [0.0 for _ in range(8)]
    for sent in split_sentences(s):
        total_cnt += 1
        predict[emotion.predict(sent)] += 1
        if emotion.predict_depression(sent) == Depression.DEPRESS:
            dep_cnt += 1

    for i in range(8):
        predict[i] = float("{:.2f}".format(predict[i] / total_cnt))
    dep_cnt = float("{:.2f}".format(dep_cnt/total_cnt))
    return predict, dep_cnt


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

#flask dev -> production 이 되려면 ewsg란 middleware를 설치해야한다.....?
#NginX -> server routing을 해주는 webserver
#flaks -> Was (WebApplicationService)

#ewsg + flask -> bentoML

#SQS -> 하나씩 처리...