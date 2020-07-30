# モジュールをインポートする
import cv2
import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import numpy as np
from datetime import datetime
import face_recognition

# 選手名
classes = ['渋野日向子', '小祝さくら', '原英莉花']
num_classes = len(classes)
image_size = 64

# アップロードされた画像を保存するファイル
UPLOAD_FOLDER = "uploads"
# アップロードを許可する拡張子を指定
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Flaskクラスのインスタンスを設定
app = Flask(__name__)

# アップロードされたファイルの拡張子をチェックする関数を定義
def allowed_file(filename):
    # １つ目の条件：変数filenameに'.'という文字が含まれているか。
    # ２つ目の条件：変数filenameの.より後ろの文字列がALLOWED_EXTENSIONSのどれに該当するかどうか
    # rsplitは区切る順序が文字列の最後から’１’回区切る。lowerは文字列を小文字に変換
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 顔を検出する(haarcascade)
def detect_face(img_path):
    image = face_recognition.load_image_file(img_path)
    faces = face_recognition.face_locations(image)
    if len(faces)>0:
        face_max = [(abs(faces[i][0]-faces[i][2])) * (abs(faces[i][1]-faces[i][3])) for i in range(len(faces))]
        top, right, bottom, left = faces[face_max.index(max(face_max))]#1人しか写っていなのでこれで問題ない
        faceImage = image[top:bottom, left:right]
        final = Image.fromarray(faceImage)

        final = np.asarray(final.resize((image_size,image_size)))
        final = Image.fromarray(final)

        basename = datetime.now().strftime("%Y%m%d-%H%M%S")
        filepath = UPLOAD_FOLDER + basename+".png"
        final.save(filepath)

        return final
    else:
        return "顔画像を入力してください"



#学習済みモデルをロードする
model = load_model('./golfer.h5', compile=False)

graph = tf.get_default_graph()

# app.route()で関数に指定したURLを対応づける。/ http://127.0.0.1:5000/以降のURLを指定
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global graph
    # as_default()で対象となるグラフを指定
    with graph.as_default():
        # HTTPメソッドがPOSTであれば
        if request.method == 'POST':
            # POSTリクエストにファイルデータが含まれているか
            if 'file' not in request.files:
                flash('ファイルがありません')
                # redirectは引数のURLに移動する関数
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            
            # ファイルがあって許可された形式であれば
            if file and allowed_file(file.filename):
                # ファイル名に危険な文字列がある場合、無効化する。
                filename = secure_filename(file.filename)
                # uploadsフォルダーに保存する
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                # ファイルパスを作成する
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                # #受け取った画像を読み込み、np形式に変換
                img = image.load_img(filepath, grayscale=False, target_size=(image_size,image_size))
                # 顔部分を検出する
                img = detect_face(filepath)

                if type(img)!=str:
                    img = image.img_to_array(img)
                    data = np.array([img])
                    #変換したデータをモデルに渡して予測する
                    result = model.predict(data)[0]
                    predicted = result.argmax()
                    pred_answer = "この女子プロは " + str(classes[predicted]) + " です"

                    return render_template("index.html",answer=pred_answer)
                else:
                    return render_template("index.html",answer=img)


        return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)