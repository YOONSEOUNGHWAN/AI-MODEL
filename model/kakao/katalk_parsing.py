import re
import pandas as pd
from model.emotion import service as emotion
from util.emotion import Emotion

Emotion = Emotion()

def katalk_msg_parse(file_path, nickname):
    my_katalk_data = list()
    # katalk_msg_pattern = "[0-9]{4}[년.] [0-9]{1,2}[일.] [0-9]{1,2}[오\S.] [0-9]{1,2}:[0-9]{1,2},.*:" #카카오톡 메시지 패턴(ios)
    katalk_msg_pattern = "[0-9]{4}[년.] [0-9]{1,2}[월.] [0-9]{1,2}[일.] 오\S [0-9]{1,2}:[0-9]{1,2},.*:"  # 카카오톡 메시지 패턴(android)

    # date_info = "[0-9]{4}년 [0-9]{1,2}월 [0-9]{1,2}일 \S요일" #날짜 바뀌는 패턴(ios)
    date_info = "[0-9]{4}년 [0-9]{1,2}월 [0-9]{1,2}일 오\S [0-9]{1,2}:[0-9]{1,2}$"  # 날짜 바뀌는 패턴(android)

    in_out_info = "[0-9]{4}[년.] [0-9]{1,2}[월.] [0-9]{1,2}[일.] 오\S [0-9]{1,2}:[0-9]{1,2}:.*"  # 나갔다는 알림 무시
    money_text = "[0-9]{4}[년.] [0-9]{1,2}[일.] [0-9]{1,2}[오\S.] [0-9]{1,2}:[0-9]{1,2},.*:.[0-9]{1,3},.*원"  # 카카오페이 돈거래
    audio_visual_text = "^동영상$|^사진$|^사진 [0-9]{1,2}장$|^이모티콘$|^봉투가 도착했어요.$|^삭제된 메시지입니다.&"  # 사진이나 동영상 메세지, 이모티콘은 추가 x

    for line in open(file_path):
        if re.match(date_info, line) or re.match(in_out_info, line) or re.match(money_text, line):
            continue
        elif line == '\n':
            continue
        elif re.match(katalk_msg_pattern, line):
            line = line.split(",")  # ,기준 2020. 1. 23. 11:57, 윤승환 : 너 친구약속 만나?
            date_time = line[0]
            user_text = line[1].split(" : ", maxsplit=1)
            user_name = user_text[0].strip()
            text = user_text[1].strip()
            if re.match(audio_visual_text, text):  # 사진 짤
                continue
            elif my_katalk_data and my_katalk_data[-1]['user_name'] == user_name:  # 동일 인물의 발화에 대해서..
                my_katalk_data[-1]['text'] += " " + text
                my_katalk_data[-1]['len'] += len(text)
            else:
                my_katalk_data.append({"date_time": date_time,
                                       "user_name": user_name,
                                       "text": text,
                                       "len": len(text)})
        else:
            if len(my_katalk_data) > 0:
                my_katalk_data[-1]['text'] += "\n" + line.strip()  # 의도적으로 문장을 나눈 경우

    my_katalk_df = pd.DataFrame(my_katalk_data)

    return customizing(my_katalk_df, nickname)

def customizing(data, nickname):
    #15글자 이하 문장 버리기
    find_index = data[data['len']<15].index
    filtering = data.drop(find_index)

    #나의 발언만 가져오기
    my_n_index = filtering[filtering['user_name'] != nickname].index
    filtering = filtering.drop(my_n_index)

    #문장 길이 삭제
    filtering = filtering.drop(['len'], axis=1)

    #감정 index생성
    filtering[['Emotion']] = filtering.apply(add_emotion, axis=1)
    return filtering.to_json(orient="records", indent=4)


def add_emotion(data):
    emotion_type = emotion.predict(data[2])
    result = Emotion.to_string(emotion_type)
    return pd.Series([result])

