from nltk.metrics import edit_distance
from PIL import Image
import pytesser
import subprocess
import pyttsx
import numpy as np
import cv2
from threading import Thread,Event
import Queue 
import os
import enchant
import goslate
from gtts import gTTS
import mp3play
import time
import math
import re
import pyaudio,wave

os.chdir(r'C:\Python27\Lib\site-packages\pytesser')
 #stream.stop_stream(),stream.close(),p.terminate()



def audio(ud):
    
     while True:
        val=ud.get()
        ud.queue.clear()
        CHUNK = 1024
        p = pyaudio.PyAudio()
        if val==True:
            wf = wave.open('toned.wav','rb')
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=wf.getnchannels(),rate=wf.getframerate(),output=True) 
            data = wf.readframes(CHUNK)
            stream.write(data)
            data = wf.readframes(CHUNK)
            stream.stop_stream()
            stream.close()
            p.terminate()
     return
    


class SpellingReplacer(object):
     def __init__(self, dict_name='en', max_dist=2):
         self.spell_dict = enchant.Dict(dict_name)
         self.max_dist = 2
     def replace(self, word):
         if self.spell_dict.check(word):
             return word
         suggestions = self.spell_dict.suggest(word)
         if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
             return suggestions[0]
         else:
             return word

def modesel():
    print "Choose a mode:\n 1.Reader\n 2.Translate\n 3.Define\n"
    while True:
        choice=input()
        if choice==1 or 2 or 3:
            break
        else:
            print "Wrong input. Enter 1,2 or 3,\n 1.Reader\n 2.Translate\n 3.Define\n"
    return choice


def translater(sent):
    sentence=''
    gs=goslate.Goslate() 
    for t in sent:
        sentence+=t
        sentence+=' '
    try:
        hindi=gs.translate(sentence,'hi')
        print hindi
        tts=gTTS(text=hindi,lang='hi')
        if tts!=None:
            tts.save('trans.mp3')
            filename = r'C:\Python27\Lib\site-packages\pytesser\trans.mp3'
            clip=mp3play.load(filename)
            clip.play()
            time.sleep(min(30, clip.seconds()))
            clip.stop()
    except Exception as ex:{}              
    return

def fingerlocker():
    
    cam=cv2.VideoCapture(1)
    lower = np.array([0,48,80])
    upper = np.array([20,255,255])
    start=time.time()
    cvals=[0]
    i=0
    cx=0
    cy=0
    print "Please keep your finger stationary......"
    while True:
            print "calculating....."
            maxcnt=0
            ret,frame=cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skinMask = cv2.inRange(hsv, lower, upper)
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
            skinMask=cv2.erode(skinMask,kernel,iterations=2)
            skinMask=cv2.dilate(skinMask,kernel,iterations=2)
            skinMask=cv2.GaussianBlur(skinMask,(1,1),0)
            skin=cv2.bitwise_and(frame,frame,mask=skinMask)
            bgr = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
            cv2.imshow('bgr',bgr)
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            contours,heirarchy=cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv2.contourArea(cnt)>maxcnt:
                    maxcnt=cv2.contourArea(cnt)
            if maxcnt>3000:
                   x,y,w,h=cv2.boundingRect(cnt)
                   cv2.rectangle(bgr,(x,y),(x+w,y+h),(0,255,0),2)
                   cv2.imshow('bgr',bgr)
                   cx=(2*x+w)/2
                   cy=y
                   av=(cx+cy)/2
                   d=cvals[i]-av
                   if abs(d)>100:
                        cvals.append(av)
                        i+=1
                        start=time.time()
                   cv2.circle(bgr,(cx,cy),3,(0,0,255),thickness=2)
                   cv2.imshow('bgr',bgr)

            stop=time.time()
            dif=stop-start
            if dif>=4 and cx!=0 and cy!=0:
                 print cx,cy
                 print "success"
                 break
            
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

    cam.release()
    cv2.destroyAllWindows()
    return (cx,cy)

def checkvalid(text):
     length=len(text)
     valid=True
     if length==1:
          if text.lower()!='a' and text.lower()!='i':
               valid=False
     
     return valid
            
def   textdet(shot,q,mode):

       sent=[]
       d=enchant.Dict("en_US")
       prev=""
       while True:
          if shot.empty()==False:
             imgarray=shot.get()
             shot.queue.clear()
             cv2.imshow('retrieved',imgarray)
             cv2.imwrite("image.jpeg" , imgarray)
             text = pytesser.image_file_to_string("image.jpeg")
             text=text.replace("\n","")
##             text=text.replace(" ","")
             for x in text:
                  if not re.match("[a-zA-Z]",x):
                     text=text.replace(x,"")
             valid=checkvalid(text)
             if text!="" and valid==True:
                   if d.check(text)==True:
                          if text.lower()!=prev:
                               print (text),
                               if mode==1:
                                   q.put(text)
                               elif mode==2:
                                   sent.append(text)
     ##                              if text[0].isupper():
                                   print sent
                                   translater(sent)
                                   del sent[:]
                          prev=text.lower()
##                   else:
##                          rep=SpellingReplacer()
##                          text_corrected=rep.replace(text)
##                          if d.check(text_corrected)==True:
##                                 print "corrected text = ",text_corrected
##                                 if mode==1:
##                                      q.put(text)
##                                 if mode==2:
##                                      sent.append(text)
##                                      print sent
##                                      translater(sent)
##                                      del sent[:]       
                          
             k = cv2.waitKey(5) & 0xFF
             if k == 27:
                    break
       cv2.destroyAllWindows() 


def speakword(q):
     engine = pyttsx.init()
     while True:
        if q.empty()==False:
             word=q.get()
             q.queue.clear()
             rate=engine.getProperty('rate')
             engine.setProperty('rate',rate-100)
             engine.say(word)
             engine.runAndWait()

def eolcheck(frame,sx,sy,mval,countleft,countright,eol,centres,bottom_limit,rx,ry,ht,ud):
     
                   if mval<sx and mval!=-1 and countleft!=0 and countright==0: 
                        eol=True
                        low=481
                        for vals in centres:
                             if vals[1]<low and vals[1]>bottom_limit:
                                  low=vals[1]
                                  rx=vals[0]
                                  ry=vals[1]
                                  ht=vals[3]
                        if rx!=0 and ry!=0:          
                             cv2.line(frame,(rx,ry),(sx,sy),(255,0,0),2)
                             cv2.imshow('frame',frame)
                        if ht!=0 and ry!=0 and rx!=0:
                             if sy>(ry+ht/2):
                                         print "GO UP  ----- END OF LINE"
                             elif sy<(ry-ht/2):
                                          print "GO DOWN ----- END OF LINE"
                   else:
                        eol=False
                   return eol,rx,ry,ht

def updowncheck(top_limit,bottom_limit,sx,sy,ud):
     
      if top_limit!=0 and bottom_limit!=0:
                   if sy>bottom_limit:
                             ud.put(True)
                   elif sy<top_limit:
                             ud.put(True)
      return

def solcheck(frame,sx,sy,minval,countleft,countright):
     if sx<minval and minval!=650 and countleft==0 and countright!=0:
          sol=True
          print "START OF LINE"
     else:
          sol=False
     return sol

def persptr(imx,pts1,pts2):
     M = cv2.getPerspectiveTransform(pts1,pts2)
     dst = cv2.warpPerspective(imx,M,(200,200))
     return dst
     
def calculateri(shot,sx,sy,ud):


     cam = cv2.VideoCapture(1)
     top_limit,bottom_limit=0,0
     rx,ry,ht=0,0,0
     eol,sol=False,False
     working=True
     while True:
        
          ret,frame=cam.read()
          cv2.imshow('frame',frame)
          blur = cv2.blur(frame,(5,5))
          gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
          ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
          thresh_2=thresh.copy()
          edges = cv2.Canny(thresh,20,40,apertureSize = 3)

   
          if np.all(edges==0)==False:

              lines = cv2.HoughLinesP(edges,1,np.pi/180,6,minLineLength=6,maxLineGap=22)
              if lines!=None:
                     for x1,y1,x2,y2 in lines[0]:
                           cv2.line(thresh,(x1,y1),(x2,y2),(0,0,0),2)        
      

              thresh_not = cv2.bitwise_not(thresh)
              contours, hierarchy = cv2.findContours(thresh_not,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
             
              mval=-1
              minval=650
##              for cnt in contours:
##                  x,y,w,h=cv2.boundingRect(cnt)
##                  if y>(sy-120) and (y+h)<(sy+120):
##                      if (x+w)>mval:
##                          mval=x+w
              MIN = np.array([0, 0, 0], np.uint8)
              MAX = np.array([100, 100, 100], np.uint8)
              cropleft=frame[sy-60:sy+60, sx-100:sx]
              cropright=frame[sy-60:sy+60, sx:sx+100]
              dstleft=cv2.inRange(cropleft,MIN,MAX)
              dstright=cv2.inRange(cropright,MIN,MAX)
              countleft= cv2.countNonZero(dstleft)
              countright= cv2.countNonZero(dstright)
                      
              L=sx-140
              R=sx+140
              U=sy-60
              D=sy+60


              cv2.circle(frame, (sx,sy), 3,(0,0,255), -1)
              cv2.rectangle(frame,(L,U),(R,D),(0,0,255),2)
              cv2.imshow('frame',frame)
              centres=[]
              for cnt in contours:

                  
                   area = cv2.contourArea(cnt)
                  
                   if area>5000 and area<=30000:

                        x,y,w,h = cv2.boundingRect(cnt)
                        if y>(sy-120) and (y+h)<(sy+120):
                           if (x+w)>mval:
                               mval=x+w
                           if x<minval:
                                minval=x

                        cx=(2*x+w)/2
                        cy=(2*y+h)/2
                        if cy>sy :
                             centres.append([cx,cy,w,h])

                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)     
                       
                    
                        cv2.imshow('frame',frame)
                         
                        if cx>=L and cx<=R and cy>=U and cy<=D and working==True:
                                 
                                 lt=thresh_2[y:y+h, x:x+w]
                                 imgarray = np.asarray(lt)
                                 pt1=[0,0]
                                 pt2=[w,0]
                                 pt3=[0,h]
                                 pt4=[w,h]
                                 pts1 = np.float32([pt1,pt2,pt3,pt4])
                                 pts2 = np.float32([[0,0],[200,0],[0,200],[200,200]])
                                 trx=persptr(imgarray,pts1,pts2)
                                 shot.put(imgarray)
                                 bottom_limit=y+h
                                 top_limit=y
                                 cv2.line(frame,(cx,cy),(sx,sy),(200,12,175),2)
                                 cv2.imshow('frame',frame)

                   [eol,rx,ry,ht]=eolcheck(frame,sx,sy,mval,countleft,countright,eol,centres,bottom_limit,rx,ry,ht,ud)
              if eol==False:
                   updowncheck(top_limit,bottom_limit,sx,sy,ud)

              if eol==True:
                   working=False       
                                
              if working==False:
                   sol=solcheck(frame,sx,sy,minval,countleft,countright)
                   if sol==True:
                        working=True

      
          k = cv2.waitKey(5) & 0xFF
          if k == 27:
             break
     cam.release()
     cv2.destroyAllWindows()

if __name__=='__main__':
     
    
##     sx,sy=fingerlocker()
     sx=320
     sy=240
     q=Queue.Queue()
     shot=Queue.Queue()
     ud=Queue.Queue()
     mode=modesel()
     convert= Thread(target=textdet, args=(shot,q,mode))
     convert.start()
     if mode==1:
         speak= Thread(target=speakword, args=(q,))
         speak.start()
     read= Thread(target=calculateri, args=(shot,sx,sy,ud))
     read.start()
     aud=Thread(target=audio, args=(ud,))
     aud.start()
     while True:
          k = cv2.waitKey(5) & 0xFF
          if k == 27:
                 convert.terminate()
                 speak.terminate()
                 read.terminate()
                 aud.terminate()
                 break
     cv2.destroyAllWindows()
                          
                            

                           
          
     
     
   

