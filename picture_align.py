"""
メモ
入力は画像二枚でベクトルを出力。
出力は画像とベクトルで動画を出力
ベクトルをあらかじめ保存して置く

表情変化が自然かどうか
本人ではなくなる
どんな変化をさせたいか？

髪の長さ
卒業アルバム
"""


import os#ファイルがない場合に作るため
import time#処理時間を図る
import glob#ファイルの中でfor文を回すため
import dlib#landmarkのため
import shutil#ファイルの中を消すため
from rembg import remove#背景を白にする
import base64

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from st_click_detector import click_detector
from PIL import Image,ImageDraw,ImageFilter

#ここから下はpSpなど
from argparse import Namespace
from pixel2style2pixel.scripts.align_all_parallel import align_face
from pixel2style2pixel.scripts.align_all_parallel import get_landmark
import torch
import torchvision.transforms as transforms
from pixel2style2pixel.utils import common
from pixel2style2pixel.models.psp import pSp
#sessionの作成
if 'size' not in st.session_state:
  st.session_state.size = None
if 'ckpt' not in st.session_state:
  st.session_state.ckpt = None
if 'net' not in st.session_state:
  st.session_state.net = None
if 'device' not in st.session_state:
  st.session_state.device = None
if 'old_csv' not in st.session_state:
  st.session_state.old_csv = None
if 'predictor' not in st.session_state:
   st.session_state.predictor = None
if 'detector' not in st.session_state:
   st.session_state.detector = None
if 'jikkou' not in st.session_state:
   st.session_state.jikkou = None
if 'save_csv1' not in st.session_state:
   st.session_state.save_csv1 = None
if 'save_csv2' not in st.session_state:
   st.session_state.save_csv2 = None
if "video1" not in st.session_state:
  st.session_state.video1 = None
if "video2" not in st.session_state:
  st.session_state.video2 = None
if "image_name" not in st.session_state:
  st.session_state.image_name = None
#if "image_name" not in st.session_state:
 # st.session_state.image_name = None


time_sta = time.time()#ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
experiment_type = 'ffhq_encode'
# モデルの設定
EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt",
        #"image_path": path1,
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    } 
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

if st.session_state.size is None:
  st.session_state.size = os.path.getsize(EXPERIMENT_ARGS['model_path'])
  print(str(st.session_state.size))
  if st.session_state.size < 1000000:
    raise ValueError("Pretrained model was unable to be downlaoded correctly!")
model_path = EXPERIMENT_ARGS['model_path']

if st.session_state.ckpt is None:
  st.session_state.ckpt = torch.load(model_path, map_location='cpu')
ckpt = st.session_state.ckpt

opts = ckpt['opts']
if st.session_state.net is None:
  # update the training options
  opts['checkpoint_path'] = model_path
  if 'learn_in_w' not in opts:
      opts['learn_in_w'] = False
  if 'output_size' not in opts:
      opts['output_size'] = 1024
  opts = Namespace(**opts)
  st.session_state.device = opts.device 
  st.session_state.net = pSp(opts)

net = st.session_state.net
net.eval()
net.cuda()

if st.session_state.predictor is None:
    st.session_state.predictor =  dlib.shape_predictor('pixel2style2pixel/shape_predictor_68_face_landmarks.dat')
if st.session_state.detector is None:
    st.session_state.detector = dlib.get_frontal_face_detector()


time_end = time.time()#eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
tim = time_end-time_sta
print("impor後から関数まえ"+str(tim))

#エンコーダ用,image_to_latent
def image_to_latent(img_path):
  input_image = Image.open(img_path)
  img_transforms = EXPERIMENT_ARGS['transform']
  transformed_image = img_transforms(input_image)
  w=net.encoder(transformed_image.unsqueeze(0).to("cuda").float())
  w=w+ckpt['latent_avg'].to(st.session_state.device).repeat(w.shape[0],1,1)
  return w

#デコーダ用、image_to_latent2,w_to_tensor,re_const
def image_to_latent2(img_path):
  input_image = Image.open(img_path)
  img_transforms = EXPERIMENT_ARGS['transform']
  transformed_image = img_transforms(input_image)
  w=net.encoder(transformed_image.unsqueeze(0).to("cuda").float())
  #w=w+ckpt['latent_avg'].to(opts.device).repeat(w.shape[0],1,1)
  return w

def w_to_tensor(w):
  time_sta = time.time()
  decode_tensor,result_latent=net.decoder([w],input_is_latent=True,randomize_noise=False,return_latents=True)
  decode_tensor=net.face_pool(decode_tensor)[0]
  time_end = time.time()
  print("--w_to_tensor="+str(time_end-time_sta))
  return decode_tensor

def re_const(coef2,csv_path,img_path):
  time_sta = time.time()
  uv1 = pd.read_csv(csv_path, index_col=0)
  uv2 = uv1.to_numpy()
  uv3 = torch.from_numpy(uv2)
  concept_vector = uv3.to('cuda')
  time_end = time.time()
  print("-csv to vector"+str(time_end-time_sta))
  time_sta = time.time()
  smile_vector = concept_vector * coef2
  with torch.no_grad():
    input_w = image_to_latent2(img_path)
  time_end = time.time()
  print("-reconst-image_to_latent2="+str(time_end-time_sta))
  bak_w =input_w
  time_sta = time.time()
  with torch.no_grad():
    ww = bak_w
    ww=ww+ckpt['latent_avg'].to(st.session_state.device).repeat(ww.shape[0],1,1)
    ww[0]=ww[0]+smile_vector
    result_tensor2 = w_to_tensor(ww)
    result_img2 = common.tensor2im(result_tensor2)
  time_end = time.time()
  print("-reconst-make-image = "+str(time_end-time_sta))
  return result_img2


#opencvで動画を開くためにbyteで保存する
def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

def run_alignment(image_path):
  time_sta = time.time()
  aligned_image = align_face(filepath=image_path, predictor=st.session_state.predictor,detector=st.session_state.detector)
  time_end = time.time()
  print("align_faceの時間"+str(time_end-time_sta))
  return aligned_image

def save_all_frames(cap,dir_path,basename,ext='jpg'):#動画をフレーム事に分ける
    shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    base_path = os.path.join(dir_path, basename)
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0
    while True:
        #retはbooleanで正しく読み込めたかどうか
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return


def save_all_align(dir_path,output_path):
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    align_input = dir_path
    in_path = '{}*'.format(align_input)
    img_paths =  glob.glob(in_path)
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        align_image = run_alignment(img_path)
        out_path = '{}{}'.format(output_path, img_name)
        align_image.save(out_path)
    

def save_all_csv1(alignmented_path,output_path,framesuu):#アライメント済の画像ファイル
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    print("all_csv実行")
    in_path = '{}*'.format(alignmented_path)
    img_paths =  glob.glob(in_path)
    before_path = None
    for img_path in img_paths:
        if before_path is not None:
          with torch.no_grad():
            magao_w=image_to_latent(before_path)
            smile_w=image_to_latent(img_path)
          diff1=smile_w[0]-magao_w[0]
          diff2=diff1.to('cpu')
          #print(diff2.size())
          diff3=diff2.numpy()
          csv_name = os.path.basename(before_path)
          csv_name = csv_name.replace('.jpg','.csv')
          path3 = '{}{}'.format(output_path, csv_name)
          pd.DataFrame(diff3).to_csv(path_or_buf=path3)
          print(str(path3) + "に保存しました")
        before_path = img_path#beforeが無い＝1枚目
    print("all_csv終了")

def save_all_csv2(video_path,output_path,framesuu):#アライメント済の画像ファイル
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    #2は原点からのベクトルを求める
    print("all_csv実行")
    path1 = '{}*'.format(video_path)
    img_paths =  glob.glob(path1)
    first_path = None
    for img_path in img_paths:
        if first_path is not None:
          with torch.no_grad():
            magao_w=image_to_latent(first_path)
            smile_w=image_to_latent(img_path)
          diff1=smile_w[0]-magao_w[0]
          diff2=diff1.to('cpu')
          #print(diff2.size())
          diff3=diff2.numpy()
          csv_name = os.path.basename(img_path)
          csv_name = csv_name.replace('.jpg','.csv')
          path3 = '{}{}'.format(output_path, csv_name)
          pd.DataFrame(diff3).to_csv(path_or_buf=path3)
          print(str(path3) + "に保存しました")
        else:
          first_path = img_path
    print("all_csv終了")

def save_all_csv3(video_path,output_path,framesuu):#アライメント済の画像ファイル
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    #3は原点からのベクトルで目とくちの表所変化を求める
    print("all_csv実行")
    ch_path = "./ch_align/"
    path1 = '{}*'.format(video_path)
    img_paths =  glob.glob(path1)
    first_path = None
    for img_path in img_paths:
        if first_path is not None:
          im = align_eye_mouse3(img_path,first_path)
          im.save(img_path)
          with torch.no_grad():
            magao_w=image_to_latent(first_path)
            smile_w=image_to_latent(img_path)
          diff1=smile_w[0]-magao_w[0]
          diff2=diff1.to('cpu')
          #print(diff2.size())
          diff3=diff2.numpy()
          csv_name = os.path.basename(img_path)
          csv_name = csv_name.replace('.jpg','.csv')
          path3 = '{}{}'.format(output_path, csv_name)
          pd.DataFrame(diff3).to_csv(path_or_buf=path3)
          print(str(path3) + "に保存しました")
        else:
          first_path = img_path
    print("all_csv終了")

def save_all_csv_eye(video_path,output_path,framesuu):#アライメント済の画像ファイル
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    #3は原点からのベクトルで目とくちの表所変化を求める
    print("all_csv実行")
    path1 = '{}*'.format(video_path)
    img_paths =  glob.glob(path1)
    first_path = None
    for img_path in img_paths:
        if first_path is not None:
          im = align_eye3(img_path,first_path)
          im.save(img_path)
          with torch.no_grad():
            magao_w=image_to_latent(first_path)
            smile_w=image_to_latent(img_path)
          diff1=smile_w[0]-magao_w[0]
          diff2=diff1.to('cpu')
          #print(diff2.size())
          diff3=diff2.numpy()
          csv_name = os.path.basename(img_path)
          csv_name = csv_name.replace('.jpg','.csv')
          path3 = '{}{}'.format(output_path, csv_name)
          pd.DataFrame(diff3).to_csv(path_or_buf=path3)
          print(str(path3) + "に保存しました")
        else:
          first_path = img_path
    print("all_csv終了")

def save_all_csv_mouse(video_path,output_path,framesuu):#アライメント済の画像ファイル
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    #3は原点からのベクトルで目とくちの表所変化を求める
    print("all_csv実行")
    path1 = '{}*'.format(video_path)
    img_paths =  glob.glob(path1)
    first_path = None
    for img_path in img_paths:
        if first_path is not None:
          im = align_mouse3(img_path,first_path)
          im.save(img_path)
          with torch.no_grad():
            magao_w=image_to_latent(first_path)
            smile_w=image_to_latent(img_path)
          diff1=smile_w[0]-magao_w[0]
          diff2=diff1.to('cpu')
          #print(diff2.size())
          diff3=diff2.numpy()
          csv_name = os.path.basename(img_path)
          csv_name = csv_name.replace('.jpg','.csv')
          path3 = '{}{}'.format(output_path, csv_name)
          pd.DataFrame(diff3).to_csv(path_or_buf=path3)
          print(str(path3) + "に保存しました")
        else:
          first_path = img_path
    print("all_csv終了")

def save_all_csv(input_path,output_path):
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    in_path = '{}*'.format(input_path)
    img_paths =  glob.glob(in_path)
    before_path = None
    for img_path in img_paths:
        if before_path is not None:
          with torch.no_grad():
            magao_w=image_to_latent(before_path)
            smile_w=image_to_latent(img_path)
          diff1=smile_w[0]-magao_w[0]
          diff2=diff1.to('cpu')
          #print(diff2.size())
          diff3=diff2.numpy()
          csv_name = os.path.basename(before_path)#連番の画像から取ることでcsvも連番になる
          csv_name = csv_name.replace('.jpg','.csv')
          path3 = '{}{}'.format(output_path, csv_name)
          pd.DataFrame(diff3).to_csv(path_or_buf=path3)
          print(str(path3) + "に保存しました")
        before_path = img_path#beforeが無い＝1枚目
    print("all_csv終了")



def make_new_frame(input_path,output_path):#再構成
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    print("make_new_frame開始")
    path1 = '{}*'.format('./video_csv/')
    csv_paths =  glob.glob(path1)
    img_path = input_path
    for csv_path in csv_paths:
      level = 1
      imim = re_const(level,csv_path,img_path)
      img_name = os.path.basename(csv_path)
      img_name = img_name.replace(".csv",".jpg")
      path4 = '{}{}'.format(output_path,img_name)
      #print(path2)
      imim.save(path4)
      img_path = path4
      print("一枚目保存"+str(path4)) 
    print("make_new_frame終了")


def make_new_frame2(input_path,output_path):
    shutil.rmtree(output_path)
    os.makedirs(output_path)
    #2は原点から
    print("make_new_frame開始")
    path1 = '{}*'.format('./video_csv2/')
    csv_paths =  glob.glob(path1)
    img_path = input_path

    first_im = re_const(1,"./zero_vector",img_path)
    img_name = os.path.basename(csv_paths[0])
    img_name = img_name.replace("1.csv","0.jpg")
    path4 = '{}{}'.format(output_path,img_name)
    first_im.save(path4)

    for csv_path in csv_paths:
      level = 1.1
      imim = re_const(level,csv_path,img_path)
      img_name = os.path.basename(csv_path)
      img_name = img_name.replace(".csv",".jpg")
      path4 = '{}{}'.format(output_path,img_name)
      imim.save(path4)
      #img_path = path4
      print("保存"+str(path4)) 
    print("make_new_frame終了")


def align_mouse(uploaded_file):#くち以外を白で埋めるマスク処理、大きさは256,256で変わらない
  in_im1 = Image.open(uploaded_file)
  a = (86, 168)
  b = (a[0]+80, a[1]+36)
  mask = Image.new('L', (256, 256), 0)
  draw = ImageDraw.Draw(mask)
  draw.rectangle(xy=(a,b), fill=255,outline=None)
  siro = Image.new("RGB",(256,256),(255,255,255))
  ima = Image.composite(in_im1,siro,mask)
  return ima  

def align_mouse2(uploaded_file):#くちだけの大きさに変更する
  in_im1 = Image.open(uploaded_file)
  a = (86, 168)
  b = (a[0]+80, a[1]+36)
  ima = in_im1.crop(a+b)
  return ima

def align_mouse3(smile,magao):#くちだけ後の画像
  in_im1 = Image.open(smile)
  im2 = Image.open(magao)
  c = (90, 165)#くち
  d = (c[0]+70, c[1]+40)
  mask = Image.new('L', (256, 256), 0)#黒画像を用意
  draw = ImageDraw.Draw(mask)
  draw.rectangle(xy=(c,d), fill=255,outline=None)#範囲を白で塗りつぶす
  mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
  ima = Image.composite(in_im1,im2,mask_blur)
  #１つ目を白い範囲に２つ目を黒い範囲に合成#境界をグレーにすると滑らかに合成できる
  return ima

def align_eye(uploaded_file):
  in_im1 = Image.open(uploaded_file)
  a = (77, 109)
  b = (a[0]+99, a[1]+21)
  mask = Image.new('L', (256, 256), 0)
  draw = ImageDraw.Draw(mask)
  draw.rectangle(xy=(a,b), fill=255,outline=None)
  siro = Image.new("RGB",(256,256),(255,255,255))
  ima = Image.composite(in_im1,siro,mask)
  return ima

def align_eye2(uploaded_file):#目だけの大きさに変更する
  in_im1 = Image.open(uploaded_file)
  a = (77, 109)
  b = (a[0]+99, a[1]+21)
  ima = in_im1.crop(a+b)
  return ima         

def align_eye3(smile,magao):#目だけ後の画像
  in_im1 = Image.open(smile)
  im2 = Image.open(magao)
  a = (66, 95)
  b = (a[0]+120, a[1]+40)
  mask = Image.new('L', (256, 256), 0)#黒画像を用意
  draw = ImageDraw.Draw(mask)
  draw.rectangle(xy=(a,b), fill=255,outline=None)#範囲を白で塗りつぶす
  mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
  ima = Image.composite(in_im1,im2,mask_blur)#１つ目を白い範囲に２つ目を黒い範囲に合成
  #境界をグレーにすると滑らかに合成できる
  return ima

def align_eye_mouse3(smile_path,magao_path):
  im1 = Image.open(smile_path)
  im1.save("./in/im1.jpg")
  im2 = Image.open(magao_path)
  im2.save("./in/im2.jpg")

  a = (66, 95)
  b = (a[0]+120, a[1]+40)
  mask = Image.new('L', (256, 256), 0)#黒画像を用意
  draw = ImageDraw.Draw(mask)
  draw.rectangle(xy=(a,b), fill=255,outline=None)#範囲を白で塗りつぶす
  c = (90, 165)#くち
  d = (c[0]+70, c[1]+40)
  draw.rectangle(xy=(c,d), fill=255,outline=None)#範囲を白で塗りつぶす
  mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
  mask.save('./mask.jpg')
  mask_blur.save("./mask_blur.jpg")
  
  target = cv2.imread("./in/im1.jpg",cv2.IMREAD_COLOR)
  base = cv2.imread("./in/im2.jpg",cv2.IMREAD_COLOR)
  mask_cv2 =cv2.imread("./mask.jpg",cv2.IMREAD_GRAYSCALE)
  point = (125,165)
  result3 = cv2.seamlessClone( target, base, mask_cv2,point,cv2.NORMAL_CLONE)
  cv2.imwrite("./result3.jpg",result3)
  ima = Image.composite(im1,im2,mask_blur)#１つ目を白い範囲に２つ目を黒い範囲に合成
  #境界をグレーにすると滑らかに合成できる
  
  return ima



def align_nofacial(smile_path,magao_path,predictor=st.session_state.predictor,detector=st.session_state.detector):#表情なしの画像,１番の表情＋２番の以外
  im1 = Image.open(smile_path)
  im1.save('./nofacial/im1.jpg')
  im2 = Image.open(magao_path)
  im2.save("./nofacial/im2.jpg")
  #in_im1 = Image.open(smile)
  lm = get_landmark('./nofacial/im1.jpg', predictor,detector)
  lm1 = np.array(lm[:17])
  lm2 = np.array(lm[26:23:-1])
  lm3 = lm[19:16:-1]
  points = np.r_[lm1,lm2,lm3]

  lm4 = np.array(lm[3:14])

  mask = np.full(shape=(256,256), fill_value=0, dtype='uint8')
  cv2.fillPoly(mask,pts = [lm4],color = 255)
  cv2.imwrite("./mask2.jpg",mask)
  white_img = np.full(shape=(256,256), fill_value=150, dtype='uint8')
  cv2.imwrite("./white_img.jpg",white_img)
  target = cv2.imread("./white_img.jpg",cv2.IMREAD_COLOR)
  base = cv2.imread("./nofacial/im1.jpg",cv2.IMREAD_COLOR)
  mask_cv2 =cv2.imread("./mask2.jpg",cv2.IMREAD_GRAYSCALE)
  point = (125,155)
  result3 = cv2.seamlessClone( target, base, mask_cv2,point,cv2.NORMAL_CLONE)
  cv2.imwrite("./nofacial/nofacial.jpg",result3)
  st.image(Image.open("./mask2.jpg"))
  ima = Image.open("./nofacial/nofacial.jpg")
  return ima


def make_csv(path1,path2):
    with torch.no_grad():
      magao2_w=image_to_latent(path1)
      smile2_w=image_to_latent(path2)

    # 差分をcsvファイルで出力
    diff1=smile2_w[0]-magao2_w[0]
    diff2=diff1.to('cpu')
    #print(diff2.size())
    diff3=diff2.numpy()
    csv = pd.DataFrame(diff3).to_csv()
    return csv


def run_nofacial(img_path):
  predictor = st.session_state.predictor
  detector = st.session_state.detector
  img = Image.open(img_path)
  img.save('./nofacial/run.jpg')
  filepath = './nofacial/run.jpg'
  #uploadedfileではpathの形が違うので一度saveして形を合わせる。
  lm = get_landmark(filepath, predictor,detector)
  facial = lm[:27]
  low = facial[8,1]
  hight = max(facial[19,1],facial[24,1])
  #memo 左下が0,0
  x_min = [0*256]
  x_max = [0*256]
  for y in range(256):
    if y < low or y > hight:
      continue#たて、横が範囲外
    elif():
       print("テスト")
       
       
def make_gif(input_path,output_path):
    gif_out = output_path
    gif_input = '{}*'.format('./gif_input/')
    gif_paths =  glob.glob(gif_input)
    img_list = []
    for gif_path in gif_paths:
      img_frame = Image.open(gif_path).quantize(method=0)
      print("open"+str(gif_path))
      img_list.append(img_frame)

    gif_name = "{}{}.gif".format(gif_out,os.path.basename(gif_paths[0]))
    img_list[0].save(gif_name, save_all=True, append_images=img_list[1:], optimize=False, duration=100, loop=0)
    
    return gif_name

def make_picture_frame(img_path,csv_path,skip = 1):
  time_sta:float = time.time()
  shutil.rmtree("./gif_input/")
  os.makedirs("./gif_input/")
  time_end:float = time.time()
  print("shutil + makedir time="+ str(time_end-time_sta))
  digit=3
  for i in range(1,111,skip):
    level =  i / 100
    time_sta = time.time()
    img = re_const(level,csv_path,img_path)
    time_end = time.time()
    tim = time_end-time_sta
    print("reconst_time="+str(tim))
    global all_re_const_time
    all_re_const_time += tim
    path_name = os.path.basename(csv_path)
    path_name = path_name.replace(".csv","")
    save_path = '{}_{}.{}'.format("./gif_input/"+path_name,str(i).zfill(digit),"jpg")
    img.save(save_path)  
     


def run_modnet(img_path,output_path):
  print("runmodnet開始ーー")
  input = Image.open(img_path)
  img = remove(input)
  img.save("./align_picture/img1.png")
  img = cv2.imread("./align_picture/img1.png", -1)#第２引数の-1はアルファチャンネルも含める
  index = np.where(img[:, :, 3] <= 200)#200は閾値アルファチャンネルの透過度、０が透明
  img[index] = [255, 255, 255, 255]
  cv2.imwrite(output_path, img)
  return 

def paste_to_bg(input_p,point):
  input_path = '{}*'.format(input_p)
  input_paths = glob.glob(input_path) 
  for input in input_paths:
    number = os.path.basename(input).replace(".jpg","")[-3:]
    bg_img = Image.open("./gif_background/page1_bg_"+number+".jpg")
    img = Image.open(input)
    #     背景画像.paste(貼り付ける画像(x,y))
    bg_img.paste(img,point)
    bg_img.save("./gif_background/page1_bg_"+number+".jpg")


def clone_bg(input_p):#画像の枚数の背景を作っておく
  img = Image.open("page1_background.jpg")
  shutil.rmtree("./gif_background/")
  os.makedirs("./gif_background/")
  input_path = '{}*'.format(input_p)
  input_paths =  glob.glob(input_path)
  for input in input_paths:
    path_name = os.path.basename(input)
    path_number = path_name.replace(".jpg","")[-3:]
    img.save("./gif_background/page1_bg_"+path_number+".jpg")


def save_mp4(input_path,output_path,fps=15):
  #fourcc = cv2.VideoWriter_fourcc(*'H264')
  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  #fourcc = 0x00000021
  path1 = '{}*'.format(input_path)
  img_paths =  glob.glob(path1)
  size = Image.open(img_paths[0]).size
  video = cv2.VideoWriter(output_path,fourcc , fps, size)
  for img_path in img_paths:
    img = cv2.imread(img_path)
    video.write(img)
  video.release()
      

def make_content(input_path,persent=20):
    #./demo_page1/template/
    path1 = '{}*'.format(input_path)
    img_paths =  glob.glob(path1)
    images = []
    for file in img_paths:
      with open(file, "rb") as image:
          encoded = base64.b64encode(image.read()).decode();
          images.append(f"data:image/jpeg;base64,{encoded}")
    result_str = ""
    for index in range(len(img_paths)):
      result_str += "<a href='#' id='{}'><img width='{}%' src={}></a> ".format(index+1,str(persent),images[index])
    #print(type(result_str))
    return  result_str

print("関数終了")
#-------------ここから下が本文-----------




st.set_page_config(layout="wide")
#st.set_page_config(layout="centered")
#ワイド好きじゃない
pagelist = ("エンコーダ","デコーダ","デモ画像","デモ動画")
#pagelist = ["エンコーダ","デコーダ","デモ画像","デモ画像2","デモ動画"]
#st.set_page_config(layout="wide")
st.title('--表情変化の検証用システム--')
#st.markdown("エンコード、デコード、ベクトルの計算を行うことができます")
#st.markdown("# "+ str(dlib.DLIB_USE_CUDA))
selector=st.sidebar.radio( "ページ選択",pagelist)

all_re_const_time:float = 0




if selector=="エンコーダ":
    st.title("エンコーダ")
    st.markdown("# 変化前の画像をアップロードしてください")
    
    
    
    uploaded_file_in1 = st.file_uploader("Choose an image...",key = 0 ,type=['png','jpg'])
    if uploaded_file_in1 is not None:
       st.image(uploaded_file_in1)
    uploaded_file_in2 = st.file_uploader("Choose an image...",key = 1,type=['png','jpg'])
    if uploaded_file_in2 is not None: 
      st.image(uploaded_file_in2)
    if uploaded_file_in1 is not None and uploaded_file_in2 is not None:
      #if st.button('実行'):
      genre = st.radio("どの方法でエンコードしますか？",
                       ('画像全体','くちマスク処理','目マスク処理',
                        'くちだけ画像小さく','目だけ画像小さく',
                        'くちだけ後画像','目だけ後画像',
                        '目＋くち','表情なし'))
      if genre == '画像全体':
        ima = Image.open(uploaded_file_in1)
        imb = Image.open(uploaded_file_in2)
        ima.save("./input/im1.jpg")
        imb.save("./input/im2.jpg")
        csv = make_csv("./input/im1.jpg","./input/im2.jpg")
        st.download_button(
          label = "Download data as CSV", 
          file_name = "sample.csv",
          data = csv,
          mime = 'text/csv',
          key = 17,
          )
      elif genre == 'くちマスク処理':
        time_sta = time.time()
        # 実行
        #画像をくちだけ切り取る
        ima = align_mouse(uploaded_file_in1)
        st.image(ima)
        imb = align_mouse(uploaded_file_in2)
        st.image(imb)
        ima.save("./mouse_input/im1.jpg")
        imb.save("./mouse_input/im2.jpg")
        csv = make_csv("./mouse_input/im1.jpg","./mouse_input/im2.jpg")
        time_end = time.time()
        tim = time_end-time_sta

        st.download_button(
          label = "Download data as mouseCSV", 
          file_name = "sample-mouse.csv",
          data = csv,
          mime = 'text/csv',
          key = 2,
          )
        
        st.write("エンコーダにかかった時間は："+ str(tim))
      elif genre == '目マスク処理':
        #目だけ切り取る
        imc = align_eye(uploaded_file_in1)
        st.image(imc)
        imd = align_eye(uploaded_file_in2)
        st.image(imd)
        imc.save("./eye_input/im1.jpg")
        imd.save("./eye_input/im2.jpg")
        csv2 = make_csv("./eye_input/im1.jpg","./eye_input/im2.jpg")

        st.download_button(
          label = "Download data as eyeCSV", 
          file_name = "sample-eye.csv",
          data = csv2,
          mime = 'text/csv',
          key = 13,
          )
      elif genre == 'くちだけ画像小さく':
        #マスクなし、口だけ
        imc = align_mouse2(uploaded_file_in1)
        st.image(imc)
        imd = align_mouse2(uploaded_file_in2)
        st.image(imd)
        imc.save("./mouseonly_input/im1.jpg")
        imd.save("./mouseonly_input/im2.jpg")
        csv3 = make_csv("./mouseonly_input/im1.jpg","./mouseonly_input/im2.jpg")

        st.download_button(
          label = "Download data as eyeCSV", 
          file_name = "sample-mouseonly.csv",
          data = csv3,
          mime = 'text/csv',
          key = 14,
          )
      elif genre == '目だけ画像小さく':
        imc = align_eye2(uploaded_file_in1)
        st.image(imc)
        imd = align_eye2(uploaded_file_in2)
        st.image(imd)
        imc.save("./eyeonly_input/im1.jpg")
        imd.save("./eyeonly_input/im2.jpg")
        csv4 = make_csv("./eyeonly_input/im1.jpg","./eyeonly_input/im2.jpg")

        st.download_button(
          label = "Download data as eyeCSV", 
          file_name = "sample-eyeonly.csv",
          data = csv4,
          mime = 'text/csv',
          key = 15,
          )
      elif genre == 'くちだけ後画像':
        #マスクなし、口だけ
        imc = align_mouse3(uploaded_file_in1,uploaded_file_in1)
        st.image(imc)
        imd = align_mouse3(uploaded_file_in2,uploaded_file_in1)
        st.image(imd)
        imc.save("./mousenew_input/im1.jpg")
        imd.save("./mousenew_input/im2.jpg")
        if st.button("実行"):
          csv3 = make_csv("./mousenew_input/im1.jpg","./mousenew_input/im2.jpg")

          st.download_button(
            label = "Download data as eyeCSV", 
            file_name = "sample-mousenew.csv",
            data = csv3,
            mime = 'text/csv',
            key = 16,
            )
      elif genre == '目だけ後画像':
        imc = align_eye3(uploaded_file_in1,uploaded_file_in1)
        st.image(imc)
        imd = align_eye3(uploaded_file_in2,uploaded_file_in1)
        st.image(imd)
        imc.save("./eyenew_input/im1.jpg")
        imd.save("./eyenew_input/im2.jpg")
        csv4 = make_csv("./eyenew_input/im1.jpg","./eyenew_input/im2.jpg")

        st.download_button(
          label = "Download data as eyeCSV", 
          file_name = "sample-eyenew.csv",
          data = csv4,
          mime = 'text/csv',
          key = 17,
          )
      elif genre == '目＋くち':
        imc = align_eye_mouse3(uploaded_file_in1,uploaded_file_in1)
        st.image(imc)
        imd = align_eye_mouse3(uploaded_file_in2,uploaded_file_in1)
        st.image(imd)
        imc.save("./eyenew_input/im1.jpg")
        imd.save("./eyenew_input/im2.jpg")
        csv4 = make_csv("./eyenew_input/im1.jpg","./eyenew_input/im2.jpg")

        st.download_button(
          label = "Download data as eyeCSV", 
          file_name = "sample-eye-mouse.csv",
          data = csv4,
          mime = 'text/csv',
          key = 19,
          )
        
      elif genre == '表情なし':
        imc = align_eye_mouse3(uploaded_file_in1,uploaded_file_in1)
        st.image(imc)
        imd = align_eye_mouse3(uploaded_file_in1,uploaded_file_in2)
        st.image(imd)
        st.write("実験")
        imh = align_nofacial(uploaded_file_in2,uploaded_file_in1)
        st.image(imh)
        imc.save("./eyenew_input/im1.jpg")
        imd.save("./eyenew_input/im2.jpg")
        csv4 = make_csv("./eyenew_input/im1.jpg","./eyenew_input/im2.jpg")

        st.download_button(
          label = "Download data as eyeCSV", 
          file_name = "sample-nofacial.csv",
          data = csv4,
          mime = 'text/csv',
          key = 19,
          )
        
      test = (
      """
      elif genre == '表情なし':
        imc = align_nofacial(uploaded_file_in1,uploaded_file_in2)
        st.header("img")
        st.image(imc)
        imd = align_nofacial(uploaded_file_in2,uploaded_file_in1)
        st.image(imd)
        imc.save("./nofacial/im1_out.jpg")
        imd.save("./nofacial/im2_out.jpg")
        csv4 = make_csv("./nofacial/im1_out.jpg","./nofacial/im2_out.jpg")

        st.download_button(
          label = "Download data as eyeCSV", 
          file_name = "nofacia;.csv",
          data = csv4,
          mime = 'text/csv',
          key = 19,
          )
      """)
        

      if st.button(label='アライメントの開始',key = 18):
        im = Image.open(uploaded_file_in1)
        im.save("./alignment_input/im1.jpg")
        test_img = run_alignment("./alignment_input/im1.jpg")
        st.image(test_img)


elif selector=="デコーダ":
  
    st.title("デコーダ")
    st.markdown("# csvファイルと画像をアップロードしてください")
    uploaded_file = st.file_uploader("Choose an csv...",type="csv")
    uploaded_file_image = st.file_uploader("Choose an image...",type="jpg")
    if uploaded_file is not None:
        csv_path = uploaded_file
        if (uploaded_file_image is not None) and (uploaded_file is not None):
            IMG_PATH = uploaded_file_image
            im = Image.open(IMG_PATH)
            if im.size[0] ==256:
              imim2 = re_const(1,"./zero_vector.csv",IMG_PATH)

              col1,col2 = st.columns(2)
              level = st.slider('how level you want to change',  min_value=1, max_value=200, step=1, value=100)
              level =  level / 100
              with col1:
                st.header("オリジナル画像")
                st.image(imim2,use_column_width = True)
            else:
              col1,col2 = st.columns(2)
              level = st.slider('how level you want to change',  min_value=1, max_value=200, step=1, value=100)
              level =  level / 100
              with col1:
                st.header("オリジナル画像")
                st.image(im,use_column_width = True)

            time_sta = time.time()
            imim = re_const(level,csv_path,IMG_PATH)
            time_end = time.time()
            tim = time_end-time_sta
            
            # 結果の出力
            with col2:
              st.header("変化後画像")
              st.image(imim,use_column_width=True)
            st.markdown("## time:" + str(tim))
            
            
            
                
elif selector == "デモ画像":
  #１，画像をアップロード
  #２，ベクトルを選択
  # 　必要なベクトル、画像全体、目のみ、くちのみ、目＋口、ポーズ
  #３，再構成して表示
  #st.header("このぺージはデモ画像用のページです")
  st.markdown("""## このページの使い方
### 1, 表情を選ぶ
### 2, 画像を選ぶ
### 実行ボタンを押すと画像が選んだ表情に変化するよ！！
""")
  #st.write("表情の変化を行います")
  content1 = make_content("./demo_page1/facial/",50)      
  facial = click_detector(content1)
  st.subheader(facial)
  if facial == "1":
    st.subheader("1番の「怒り」が選択されています")
    vector = "angry"
  elif facial == "2":
    st.subheader("２番の「不満」が選択されています")
    vector = "complain"
  elif facial == "3":
    st.subheader("３番の「笑顔」が選択されています")
    vector = "smile"
  st.write(" ")
  st.write(" ")
  st.write(" ")
  st.write(" ")
  st.write(" ")
  st.write(" ")
  st.markdown("  ### どの画像を変化させたいですか？")
  #input_path = '{}*'.format("./demo_page1/template/")
  #input_paths = glob.glob(input_path)
  #img_paths = [] 
  #number = 0
  #for input in input_paths:
  #   number += 1
  #   img_paths.append("./demo_page1/template/sample{}.jpg".format(str(number)))


  content = make_content("./demo_page1/template/")      
  clicked = click_detector(content)
  st.subheader(clicked)
  uploaded_file_image = st.sidebar.file_uploader("自分の画像を使いたい場合はアップロードしてください",type="jpg")
  if uploaded_file_image is not None:
    if uploaded_file_image.name is not st.session_state.image_name: 
      st.session_state.image_name = uploaded_file_image.name
      img = Image.open(uploaded_file_image)
      if 256 < img.size[0] or 256 < img.size[1]:
        img.save("./align_picture/img1.jpg")
        img = run_alignment("./align_picture/img1.jpg")
        img.save("./align_picture/img1.jpg")
        run_modnet("./align_picture/img1.jpg","./align_picture/img1.jpg")
        img = Image.open("./align_picture/img1.jpg")
      st.image(img)
  else:
     print("uploaded_file_image is none")
     if clicked in ("1","2","3","4","5","6","7","8"):
        st.image(Image.open("./demo_page1/template/sample{}.jpg".format(clicked)))
  st.write("この画像でよろしいですか？")
  skip = 3
  if st.button("表情変化を実行する"):
    time_sta:float
    time_end:float
    time_sta1:float 
    time_end1:float
    tim:float
    time_sta = time.time()
    if uploaded_file_image is not None:
      print("画像アップロードを選びました")
      img = Image.open("./align_picture/img1.jpg")
      img.save("./in/img1.jpg")
          #angry,complain,smile
      gif_f = []
      st.write("表情変化元ー作成開始ー1/6")
      make_picture_time = 0
      time_sta1 = time.time()
      make_picture_frame(img_path="./gif_{}/original.jpg".format(vector),csv_path='./demo_page1/{}.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      
      print("make_picture_frame(original)"+str(tim1))
      clone_bg("./gif_input/")
      paste_to_bg("./gif_input/",(1,104))
      st.write("表情変化先ー作成開始ー2/6")
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      print("make_picture_frame(all)"+str(tim1))
      paste_to_bg("./gif_input/",(1,443))
      st.write("Poseのみー作成開始ー3/6")
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}_nofacial.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      print("make_picture_frame(nofacial)"+str(tim1))
      paste_to_bg("./gif_input/",(371,104))
      st.write("表情のみー作成開始ー4/6")
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}_eye_mouse.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      print("make_picture_frame(face_all)"+str(tim1))
      paste_to_bg("./gif_input/",(370,443))
      st.write("目の変化のみー作成開始ー5/6")
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}_eye.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      print("make_picture_frame(eye)"+str(tim1))
      paste_to_bg("./gif_input/",(743,104))
      st.write("口の変化のみー作成開始ー6/6")
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}_mouse.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      print("make_picture_frame(mouth)"+str(tim1))
      paste_to_bg("./gif_input/",(743,443))
      time_end = time.time()
      tim = time_end-time_sta
      print("表情変化開始ー終わり＝"+tim)
      print("makepicture_frame_all="+str(make_picture_time))
      print("all_reconst_time = "+str(all_re_const_time))
      st.write("実行時間＝"+str(tim))


      save_mp4("./gif_background/","bg_test1.mp4")
      time.sleep(1)
      bg_video_file = open('bg_test1.mp4', 'rb')
      st.session_state.video1 = bg_video_bytes = bg_video_file.read()
    else :#画像がアップロードされてないとき
      print("stock = "+ clicked)
      img = Image.open("./demo_page1/template/sample{}.jpg".format(clicked))
      img.save("./in/img1.jpg")
      gif_f = []
      st.write("表情変化元ー作成開始ー1/6")
      make_picture_time:float = 0.0
      not_make_picture_time:float = 0.0
      time_sta1 = time.time()
      make_picture_frame(img_path="./gif_{}/original.jpg".format(vector),csv_path='./demo_page1/{}.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      time_sta1 = time.time()
      print("make_picture_frame(original)"+str(tim1))
      clone_bg("./gif_input/")
      paste_to_bg("./gif_input/",(1,104))
      st.write("表情変化先ー作成開始ー2/6")
      time_end1 = time.time()
      not_make_picture_time += time_end1 - time_sta1
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      time_sta1 = time.time()
      print("make_picture_frame(all)"+str(tim1))
      paste_to_bg("./gif_input/",(1,443))
      st.write("Poseのみー作成開始ー3/6")
      time_end1 = time.time()
      not_make_picture_time += time_end1 - time_sta1
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}_nofacial.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      time_sta1 = time.time()
      print("make_picture_frame(nofacial)"+str(tim1))
      paste_to_bg("./gif_input/",(371,104))
      st.write("表情のみー作成開始ー4/6")
      time_end1 = time.time()
      not_make_picture_time += time_end1 - time_sta1
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}_eye_mouse.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      time_sta1 = time.time()
      print("make_picture_frame(face_all)"+str(tim1))
      paste_to_bg("./gif_input/",(370,443))
      st.write("目の変化のみー作成開始ー5/6")
      time_end1 = time.time()
      not_make_picture_time += time_end1 - time_sta1
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}_eye.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      time_sta1 = time.time()
      print("make_picture_frame(eye)"+str(tim1))
      paste_to_bg("./gif_input/",(743,104))
      st.write("口の変化のみー作成開始ー6/6")
      time_end1 = time.time()
      not_make_picture_time += time_end1 - time_sta1
      time_sta1 = time.time()
      make_picture_frame(img_path="./in/img1.jpg",csv_path='./demo_page1/{}_mouse.csv'.format(vector),skip= skip)
      time_end1 = time.time()
      tim1 = time_end1-time_sta1
      make_picture_time += tim1
      time_sta1 = time.time()
      print("make_picture_frame(mouth)"+str(tim1))
      paste_to_bg("./gif_input/",(743,443))
      time_end1 = time.time()
      not_make_picture_time += time_end1 - time_sta1
      time_end = time.time()
      tim = time_end-time_sta
      print("表情変化開始ー終わり＝"+str(tim))
      print("all_makepicture_frame="+str(make_picture_time))
      print("not makepicture"+str(not_make_picture_time))
      print("all_reconst_time = "+str(all_re_const_time))
      st.write("実行時間＝"+str(tim))

      time_sta = time.time()
      save_mp4("./gif_background/","bg_test1.mp4")
      time.sleep(1)
      bg_video_file = open('bg_test1.mp4', 'rb')
      time_end = time.time()
      tim = time_end - time_sta
      print("makevideo = "+str(tim))
      st.session_state.video1 = bg_video_bytes = bg_video_file.read()

  if st.session_state.video1 is not None:
     st.video(st.session_state.video1)
  
  st.header("アンケートお願いします。質問は4つです")
  img = Image.open("googleform_qr.jpg")
  st.image(img.resize((290,290),resample=Image.NEAREST))




elif selector =="デモ画像2":
    #csvベクトルを選べるようにする
    #demo_page1に入っているファイルを用いる
    #怒り、
    houkou = st.radio("表情の方向",("怒り","不満","笑顔"),horizontal=True)
    #st.radio("変化する部分",("全体","表情",))
    if houkou == "怒り":
      st.image("")
    st.button("")

elif selector =="デモ動画":
    st.title("このぺージはデモ用のページです")
    st.markdown("# 動画をアップロードしてください")
    uploaded_file_mp4 = st.sidebar.file_uploader("Choose an mp4...",key = 3,type="mp4")
    uploaded_file_image2 = st.sidebar.file_uploader("Choose an image...",key = 4,type=["png","jpg"])
    if uploaded_file_mp4 :
        #if st.button(key = 8,label = "このファイルのベクトルを抽出しますか?"):
          temp_file_to_save = './temp_file_1.mp4'
          write_bytesio_to_file(temp_file_to_save, uploaded_file_mp4)
          # read it with cv2.VideoCapture(), so now we can process it with OpenCV functions
          cap = cv2.VideoCapture(temp_file_to_save)
          framesuu = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          st.write("フレーム数は",framesuu)
          video_csv_path = './video_frame/'
          if not os.path.exists(video_csv_path):
            os.makedirs(video_csv_path)
            #print("追加しました")
          if st.button(key = 5,label='frameに切り抜きします'):
            save_all_frames(cap, video_csv_path, 'sample_video_img')
          #align
          if st.button(key = 6,label='alignmentを開始します'):
            time_sta = time.time()
            save_all_align(video_csv_path,output_path='./alignment/')
            time_end = time.time()
            tim = time_end-time_sta
            st.write("alignが終了しました")
            st.write("かかった時間は"+ str(tim))
          if st.button(key = 10,label = "フレームごとのcsvファイルを作成します"):
            save_all_csv1(video_path='./alignment/',output_path='./video_csv/',framesuu=framesuu)#アライメント済の画像ファイル
            print("save_all_csvが終了しました")
          if st.button(key = 11,label = "原点からのcsvファイルを作成します"):
            save_all_csv2(video_path='./alignment/',output_path='./video_csv2/',framesuu=framesuu)#アライメント済の画像ファイル
            print("save_all_csvが終了しました")
          if st.button(key = 14,label = "目だけのcsvファイルを作成します"):
            save_all_csv_eye(video_path='./alignment/',output_path='./video_csv2/',framesuu=framesuu)#アライメント済の画像ファイル
            print("save_all_csvが終了しました")
          if st.button(key = 17,label = "くちだけのcsvファイルを作成します"):
            save_all_csv_mouse(video_path='./alignment/',output_path='./video_csv2/',framesuu=framesuu)#アライメント済の画像ファイル
            print("save_all_csvが終了しました")
          if st.button(key = 18,label = "目とくちだけのcsvファイルを作成します"):
            save_all_csv3(video_path='./alignment/',output_path='./video_csv2/',framesuu=framesuu)#アライメント済の画像ファイル
            print("save_all_csvが終了しました")
          
          if uploaded_file_image2:
            if st.button(key = 7,label='フレームごとの再構成を開始します'):
              make_new_frame(input_path=uploaded_file_image2,output_path='./new_frame/')
            if st.button(key = 9,label='原点からの再構成を開始します'):
              make_new_frame2(input_path=uploaded_file_image2,output_path='./new_frame/')
          if st.button(key = 12,label="動画を作成します"):
            save_mp4("./new_frame/","douga_test1.mp4",29)
            time.sleep(1)
            bg_video_file = open('douga_test1.mp4', 'rb')
            st.session_state.video2 = bg_video_bytes = bg_video_file.read()
          acol1,acol2,acol3 = st.columns(3)
          with acol2:
            if st.session_state.video2 is not None:
              st.video(st.session_state.video2)

              

          