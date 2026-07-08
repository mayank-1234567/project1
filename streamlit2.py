from flask import Flask, request, jsonify, render_template
import torch.nn as nn
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
import uuid
from io import BytesIO
import numpy as np
import cv2
import base64
import pymupdf
piece_class=["k","q","r","b","n","p","K","Q","R","B","N","P","empty"]

class squareNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*12*12,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,13)
        )
    def forward(self,x):
            x=self.features(x)
            x=self.classifier(x)
            return x

model = squareNet()
MODEL_PATH = Path(__file__).with_name("model_best.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()
transform = transforms.Compose([transforms.ToTensor()])
###model space


def board_crop(image):
    image=np.array(image)
    ####takes an image and returns a list of cropped images of the board
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    gray=cv2.equalizeHist(blur)
    thre4=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours,heir= cv2.findContours(thre4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours1=sorted(contours,key=cv2.contourArea,reverse=True)
    approx=[]
    X=[]
    Y=[]
    W=[]
    H=[]
    cropped=[]
    for cnt in contours1:
        epsilon=0.02*cv2.arcLength(cnt,True)
        rect=cv2.approxPolyDP(cnt,epsilon=epsilon,closed=True)
        area=cv2.contourArea(rect)
        if len(rect)==4 and area>30000 :
           approx.append(rect)
           x,y,w,h=cv2.boundingRect(rect)
           if x!=0 and y!=0:
                X.append(x)
                Y.append(y)
                H.append(h)
                W.append(w)
                
    for i in range(len(X)):
            
            cropped.append(image[Y[i]:Y[i]+H[i],X[i]:X[i]+W[i]])

         
        

    return cropped
def square_crop(cropped):
        ###takes a list of cropped images and returns a list of cropped images of the squares
        squares=[]
        cropped = Image.fromarray(cropped)
        crop_r=cropped.resize((800,800),Image.LANCZOS)
        sw=100
        sh=100
        for y in range(8):
            for x in range(8):
               square=crop_r.crop((x*sw,y*sh,(x+1)*sw,(y+1)*sh))
               squares.append(square)
        return squares

 

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("new1.html")
    

pdf_store={}
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    
    data = request.json

    pdf_base64 = data["pdf"]
    
    # Decode Base64 PDF
    pdf_bytes = base64.b64decode(pdf_base64)
    pdf_id = str(uuid.uuid4())
    pdf_store[pdf_id] = pdf_bytes
    print(f"PDF stored with ID: {pdf_id}, Size: {len(pdf_bytes)} bytes, Type: {type(pdf_store[pdf_id])}")
    # Open PDF from memory
    docim = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    
    


    fent=[]
    with torch.inference_mode():
        
            page=docim.load_page(0)

            pix=page.get_pixmap(matrix=pymupdf.Matrix(2,2))
            img_data=BytesIO(pix.tobytes())
            image=Image.open(img_data).convert("RGB")
            cropped=board_crop(image)
            for i in range(len(cropped)):
                r=""
                d=0
                if len(cropped)==0:
                    print("no board found")
                    continue
                else:
                    squares=square_crop(cropped[i])
                    for j in range(1,len(squares)+1):
                        square=squares[j-1]
                        square=transform(square)
                        square=square.unsqueeze(0)
                        output=model(square)
                        predicted_class=torch.argmax(output,dim=1).item()
                        
                        if predicted_class==12:
                            
                            d+=1
                        else:
                            if d>0:
                                r+=str(d)
                                d=0
                            r+=piece_class[predicted_class]
                        if j==len(squares):
                           if d > 0:
                                 r += str(d)
                        if j%8==0:
                            r+="/"
                        
                    
                r=r.rstrip("/")
                fen = r

                
                fent.append(fen)
    docim.close()
    if len(fent)==0:
        return jsonify(fen=[], pdf_id=pdf_id)
    return jsonify(fen=fent, pdf_id=pdf_id)
@app.route("/page_count", methods=["POST"])
def page_count():
    
    data = request.json
    page_count=data["page_count"]
    pdf_id=data["pdf_id"]
    pdf_bytes = pdf_store.get(pdf_id)
    pdf_bytes = pdf_store[pdf_id]

    print(type(pdf_bytes))
    print(len(pdf_bytes))
    fent=[]
    docim = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    with torch.inference_mode():
        
            page=docim.load_page(page_count-1)

            pix=page.get_pixmap(matrix=pymupdf.Matrix(2,2))
            img_data=BytesIO(pix.tobytes())
            image=Image.open(img_data).convert("RGB")
            cropped=board_crop(image)
            for i in range(len(cropped)):
                r=""
                d=0
                if len(cropped)==0:
                    print("no board found")
                    continue
                else:
                    squares=square_crop(cropped[i])
                    for j in range(1,len(squares)+1):
                        square=squares[j-1]
                        square=transform(square)
                        square=square.unsqueeze(0)
                        output=model(square)
                        predicted_class=torch.argmax(output,dim=1).item()
                        
                        if predicted_class==12:
                            
                            d+=1
                        else:
                            if d>0:
                                r+=str(d)
                                d=0
                            r+=piece_class[predicted_class]
                        if j==len(squares):
                           if d > 0:
                                 r += str(d)
                        if j%8==0:
                            r+="/"
                r=r.rstrip("/")
                fen = r

                
                fent.append(fen)
    docim.close()
    if len(fent)==0:
        return jsonify(fen=[], pdf_id=pdf_id)
    return jsonify(fen=fent, pdf_id=pdf_id)
if __name__ == "__main__":
    app.run(debug=False)