# モジュールをインポートする
import cv2
import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np

# 選手名
# classes = ['渋野日向子', '大里桃子', '河本', '小祝さくら']
classes = ['小祝さくら', '河本', '渋野日向子', '大里桃子']
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

def detect_face(img_path):
    img = cv2.imread(img_path, 1)
    
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_file = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_file)
    faces = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=1, minSize=(10,10))
    if len(faces)>0:
        i = 0
        #顔の座標を取り出す
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            img = cv2.resize(face, (64,64))
            return img
    else:
        return "顔画像を入力してください"

#学習済みモデルをロードする
model = load_model('./golfer_4.h5', compile=False)

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

                img = image.img_to_array(img)
                data = np.array([img])
                #変換したデータをモデルに渡して予測する
                result = model.predict(data)[0]
                predicted = result.argmax()
                pred_answer = "これは " + str(classes[predicted]) + " 選手です"

                return render_template("index.html",answer=pred_answer)

        return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)

# https://jagirl-golfer.herokuapp.com/