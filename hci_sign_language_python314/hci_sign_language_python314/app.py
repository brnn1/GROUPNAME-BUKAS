import cv2, tkinter as tk, time, math
from tkinter import ttk, messagebox
from collections import deque

ACTIONS = {"OPEN_PALM":"Pause / Stop","FIST":"Play / Resume","POINT":"Next slide","PEACE":"Previous slide","THUMBS_UP":"Volume up"}

def hand_contour(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 25, 45), (25, 180, 255))
    m2 = cv2.inRange(hsv, (160, 25, 45), (180, 180, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    h,w = mask.shape
    x1,y1,x2,y2 = int(w*.2), int(h*.08), int(w*.88), int(h*.94)
    roi = mask[y1:y2,x1:x2]
    contours,_ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, (x1,y1,x2,y2)
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < w*h*.012: return None, (x1,y1,x2,y2)
    return c + (x1,y1), (x1,y1,x2,y2)

def classify(c):
    if c is None: return "NO_HAND"
    area=cv2.contourArea(c); per=cv2.arcLength(c,True)
    if area <= 0 or per <= 0: return "UNKNOWN"
    hull=cv2.convexHull(c); ha=max(cv2.contourArea(hull),1)
    x,y,w,h=cv2.boundingRect(c)
    solidity=area/ha; circularity=4*math.pi*area/(per*per)
    defects_n=0
    try:
        hi=cv2.convexHull(c,returnPoints=False)
        if hi is not None and len(hi)>3:
            ds=cv2.convexityDefects(c,hi)
            if ds is not None:
                defects_n=sum(1 for d in ds[:,0] if d[3]/256 > max(12,h*.045))
    except cv2.error: pass
    if defects_n >= 3: return "OPEN_PALM"
    if defects_n == 2: return "PEACE"
    if defects_n == 1: return "POINT"
    if solidity > .82 and circularity > .30: return "FIST"
    if w < h*.72: return "THUMBS_UP"
    return "UNKNOWN"

class App:
    def __init__(self, root):
        self.root=root; root.title("Human-Computer Interaction & Sign Language")
        root.geometry("1080x700"); root.minsize(850,600)
        self.cam=None; self.running=False; self.mode=tk.StringVar(value="Gesture Control")
        self.pred=tk.StringVar(value="Prediction: —"); self.action=tk.StringVar(value="Suggested action: —")
        self.status=tk.StringVar(value="Click Start Camera.")
        self.history=deque(maxlen=5); self.last_label=""; self.last_time=0
        outer=ttk.Frame(root,padding=14); outer.pack(fill="both",expand=True)
        ttk.Label(outer,text="Human-Computer Interaction & Sign Language",font=("Segoe UI",17,"bold")).pack(anchor="w")
        ttk.Label(outer,text="Python 3.14 edition • OpenCV webcam prototype").pack(anchor="w",pady=(4,12))
        body=ttk.Frame(outer); body.pack(fill="both",expand=True)
        left=ttk.Frame(body); left.pack(side="left",fill="both",expand=True)
        right=ttk.Frame(body,width=280); right.pack(side="right",fill="y",padx=(14,0)); right.pack_propagate(False)
        self.video=ttk.Label(left,text="Camera stopped",anchor="center"); self.video.pack(fill="both",expand=True)
        ttk.Label(left,text="Use good lighting and keep your hand in the outlined area.").pack(anchor="w",pady=8)
        ttk.Label(right,text="Operating mode",font=("Segoe UI",11,"bold")).pack(anchor="w")
        combo=ttk.Combobox(right,textvariable=self.mode,state="readonly",values=["Gesture Control","Sign Language (not trained)"])
        combo.pack(fill="x",pady=6); combo.bind("<<ComboboxSelected>>",self.mode_changed)
        ttk.Label(right,textvariable=self.pred,font=("Segoe UI",13,"bold"),wraplength=260).pack(anchor="w",pady=(12,8))
        ttk.Label(right,textvariable=self.action,wraplength=260).pack(anchor="w",pady=6)
        ttk.Button(right,text="Start Camera",command=self.start).pack(fill="x",pady=4)
        ttk.Button(right,text="Stop Camera",command=self.stop).pack(fill="x",pady=4)
        ttk.Button(right,text="How to test",command=self.help).pack(fill="x",pady=4)
        ttk.Separator(right).pack(fill="x",pady=12)
        ttk.Label(right,text="Gesture mapping",font=("Segoe UI",11,"bold")).pack(anchor="w")
        ttk.Label(right,text="Open palm → Pause / Stop\nFist → Play / Resume\nPoint → Next slide\nPeace → Previous slide\nThumbs up → Volume up",justify="left").pack(anchor="w",pady=6)
        ttk.Label(right,textvariable=self.status,wraplength=260).pack(anchor="w",pady=10)
        root.protocol("WM_DELETE_WINDOW",self.close)
    def help(self):
        messagebox.showinfo("Test the system","1. Click Start Camera.\n2. Keep one hand centered in the box.\n3. Try open palm, fist, pointing, and peace sign.\n\nThis Python 3.14 edition uses basic OpenCV skin-color segmentation. Results depend on lighting and background. Sign-language recognition needs a separate trained model; this build does not pretend to translate signs.")
    def mode_changed(self,_=None):
        if self.mode.get().startswith("Sign"):
            self.status.set("Sign recognition is not trained in this Python 3.14 edition.")
        else: self.status.set("Gesture Control selected.")
    def start(self):
        if self.running: return
        self.cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            self.cam.release(); self.cam=cv2.VideoCapture(0)
        if not self.cam.isOpened():
            self.status.set("Camera unavailable. Close other camera apps or check Windows camera permissions."); return
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,640); self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.running=True; self.status.set("Camera running."); self.update()
    def update(self):
        if not self.running or self.cam is None: return
        ok,frame=self.cam.read()
        if not ok:
            self.status.set("Cannot read camera frame."); self.root.after(100,self.update); return
        frame=cv2.flip(frame,1); contour,roi=hand_contour(frame); x1,y1,x2,y2=roi
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,180,0),1)
        if contour is not None:
            cv2.drawContours(frame,[contour],-1,(0,255,100),2)
            x,y,w,h=cv2.boundingRect(contour); cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,100),2)
        label=classify(contour) if self.mode.get()=="Gesture Control" else "MODEL_NOT_TRAINED"
        self.history.append(label); stable=max(set(self.history),key=self.history.count)
        cv2.putText(frame,stable.replace("_"," "),(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(40,240,100),2)
        self.pred.set("Prediction: "+stable.replace("_"," "))
        self.action.set("Suggested action: "+ACTIONS.get(stable,"No mapped action"))
        if stable in ACTIONS and stable != self.last_label and time.time()-self.last_time>1:
            self.last_label=stable; self.last_time=time.time()
            self.status.set("Detected "+stable.replace("_"," ")+". Suggested action only; no keys are sent.")
        # Tkinter can display webcam images using Pillow.
        try:
            from PIL import Image, ImageTk
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            image.thumbnail((max(400,self.video.winfo_width()),max(300,self.video.winfo_height())))
            photo=ImageTk.PhotoImage(image); self.video.configure(image=photo,text=""); self.video.image=photo
        except ImportError:
            self.status.set("Pillow missing. Install with: python -m pip install pillow")
        self.root.after(25,self.update)
    def stop(self):
        self.running=False
        if self.cam is not None: self.cam.release(); self.cam=None
        self.video.configure(image="",text="Camera stopped"); self.video.image=None
        self.status.set("Camera stopped.")
    def close(self): self.stop(); self.root.destroy()

if __name__=="__main__":
    root=tk.Tk(); App(root); root.mainloop()
