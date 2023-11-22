from flask import Flask, jsonify, request
from gensim.models import doc2vec
import MakeDB
import InferVector

data = MakeDB.run()
app = Flask(__name__)


json = dict()
json['name'] = '논문'
json['abstract'] = 'abstract'
json['author'] = '저자'
json['year'] = '출판년도'
json['link'] = '논문링크'
json['category'] = '분야'
'''
{
    "name" : "한국어교육에서의 대화형 인공지능 챗봇 적용 가능성 탐색 -고급 한국어 학습자와 ChatGPT의 상호작용 분석을 중심으로-",
    "abstract" : "",
    "author" : "",
    "year" : "",
    "link" : "",
    "category" : "인문학_국어국문학"
}
'''


@app.route('/test', methods=['POST'])
def post2():
    #여러개의 json을 받아오는 경우
    params = request.get_json()
    category = ''

    text = []
    name_list = []
    for json in params:
        name = json['name']
        category = json['category']
        print(json)
        text.append(data[name][1])
        name_list.append(name)

    model_path = category.split('_')[0]
    model_name = category + '_한국어_model'
    model = doc2vec.Doc2Vec.load('../API/Model/{}/{}'.format(model_path, model_name))
    # 생성한 모델 로드
    infer_vec = InferVector.infer_vector(text, model)
    similar_doc = model.dv.most_similar(positive=[infer_vec], topn=20)
    # 5개까지 추천해주도록 설정

    json_obj = []
    count = 0
    for doc_name, cosine in similar_doc:
        if doc_name in name_list: #기존에 사용자가 서재에 추가한 논문의 경우는 추천 논문 리스트에 포함시키지 않음
            continue;
        count+=1
        json = dict()
        json['name'] = doc_name
        json['abstract'] = data[doc_name][1]
        json['author'] = data[doc_name][2]
        json['year'] = data[doc_name][3]
        json['link'] = data[doc_name][4]
        json['category'] = data[doc_name][5]
        json_obj.append(json)
        if count == 5:
            break

    return jsonify(json_obj)


@app.route('/post', methods=['POST'])
def post():
    params = request.get_json()
    name = params['name']
    category = params['category']
    #json을 받아와서, 해당 논문의 이름과 분야를 받아온다

    model_path = category.split('_')[0]
    model_name = category + '_한국어_model'
    #print(model_name, model_path)
    model = doc2vec.Doc2Vec.load('../API/Model/{}/{}'.format(model_path, model_name))
    #생성한 모델 로드

    similar_doc = model.dv.most_similar(name, topn=5)
    #5개까지 추천해주도록 설정

    json_obj = []
    for doc_name, cosine in similar_doc:
        json = dict()
        json['name'] = doc_name
        json['abstract'] = data[doc_name][1]
        json['author'] = data[doc_name][2]
        json['year'] = data[doc_name][3]
        json['link'] = data[doc_name][4]
        json['category'] = data[doc_name][5]
        json_obj.append(json)
    return jsonify(json_obj)


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)