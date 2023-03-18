import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import re
from PIL import Image 
import torch 
from torch import autocast 
from diffusers import ( 
    DDIMScheduler, 
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline
) 

class StableDiffusion(tk.Tk):
    MODEL_ID = "CompVis/stable-diffusion-v1-4"
    # MODEL_ID = "stabilityai/stable-diffusion-2"
    # DEVICE = "cuda"
    DEVICE = "cpu"
    YOUR_TOKEN = "トークン"    # https://huggingface.co/joinのアクセストークン
    INPUT_IMG = "input.jpg"
    OUTPUT_IMG = "output.jpg"
    PROMPT = "A fantasy landscape, trending on artstation"
    SEED = 1024
    STEP = 100
    SCALE = 7.5
    STRENGTH = 0.8
    WIDTH = 512
    HEIGHT = 512

    def __init__(self):
        super().__init__()
        self.title("Todo App")
        self.geometry("800x400")

        # ラベルとエントリーを配置するフレーム
        frame = tk.Frame(self)
        frame.pack(pady=10)
        tk.Label(frame, text="INPUT_IMG:", width=10).grid(row=0, column=0)
        self.lbl_input_img = tk.Label(frame, text=self.INPUT_IMG, width=50, anchor="w")
        self.lbl_input_img.grid(row=0, column=1)
        self.bt_input_image = tk.Button(frame, text="input image", command=self.inputImage)
        self.bt_input_image.grid(row=0, column=2)
        self.bt_clear = tk.Button(frame, text="clear", command=self.clearInputImage)
        self.bt_clear.grid(row=0, column=3)
        
        tk.Label(frame, text="OUTPUT_IMG:", width=10).grid(row=1, column=0)
        self.lbl_output_img = tk.Label(frame, text=self.OUTPUT_IMG, width=50, anchor="w")
        self.lbl_output_img.grid(row=1, column=1)
        self.bt_output_image = tk.Button(frame, text="output image", command=self.outputImage)
        self.bt_output_image.grid(row=1, column=2, columnspan=2)

        tk.Label(frame, text="WIDTH:", width=10).grid(row=2, column=0)
        self.width_text = tk.Entry(frame, width=60)
        self.width_text.insert(0, self.WIDTH)
        self.width_text.grid(row=2, column=1, columnspan=3)

        tk.Label(frame, text="HEIGHT:", width=10).grid(row=3, column=0)
        self.height_text = tk.Entry(frame, width=60)
        self.height_text.insert(0, self.HEIGHT)
        self.height_text.grid(row=3, column=1, columnspan=3)

        tk.Label(frame, text="STRENGTH:", width=10).grid(row=4, column=0)
        self.strength_text = tk.Entry(frame, width=60)
        self.strength_text.insert(0, self.STRENGTH)
        self.strength_text.grid(row=4, column=1, columnspan=3)

        tk.Label(frame, text="PROMPT:", width=10).grid(row=5, column=0)
        self.prompt_text = tk.Entry(frame, width=60)
        self.prompt_text.insert(0, self.PROMPT)
        self.prompt_text.grid(row=5, column=1, columnspan=3)

        stableDiffusion_frame = tk.Frame(self)
        stableDiffusion_frame.pack(pady=10)
        stable_img2img_btn = tk.Button(stableDiffusion_frame, text="create image To image", command=self.img2img)
        #stable_img2img_btn.pack(side="left")
        stable_img2img_btn.grid(row=0, column=0)
        stable_txt2img_btn = tk.Button(stableDiffusion_frame, text="create text To image", command=self.txt2img)
        #stable_txt2img_btn.pack(side="left")
        stable_txt2img_btn.grid(row=0, column=1)

    def onChangeWidthText(self):
        try:
            self.WIDTH = int(self.width_text.get())

            if self.WIDTH < 0:
                self.WIDTH = 0
            
            self.width_text.delete(0, tk.END)
            self.width_text.insert(0, self.WIDTH)
        except:
            self.WIDTH = 0
            self.width_text.delete(0, tk.END)
            self.width_text.insert(0, self.WIDTH)

    def onChangeHeightText(self):
        try:
            self.HEIGHT = int(self.width_text.get())

            if self.HEIGHT < 0:
                self.HEIGHT = 0

            self.height_text.delete(0, tk.END)
            self.height_text.insert(0, self.HEIGHT)
        except:
            self.HEIGHT = 0
            self.height_text.delete(0, tk.END)
            self.height_text.insert(0, self.HEIGHT)

    def onChangeStrength(self):
        try:
            self.STRENGTH = float(self.strength_text.get())

            if self.STRENGTH > 1:
                self.STRENGTH = 1.0
            elif self.STRENGTH < 0:
                self.STRENGTH = 0.0

            self.strength_text.delete(0, tk.END)
            self.strength_text.insert(0, self.STRENGTH)
        except:
            self.STRENGTH = 0
            self.strength_text.delete(0, tk.END)
            self.strength_text.insert(0, self.STRENGTH)

    def onChangePrompt(self):
        self.PROMPT = self.prompt_text.get()

    def inputImage(self):
        # typ = [('テキストファイル','*.txt')] 
        # fle = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
        fle = filedialog.askopenfilename() 

        self.INPUT_IMG = fle
        self.lbl_input_img["text"] = self.INPUT_IMG

    def clearInputImage(self):
        self.INPUT_IMG = ''
        self.lbl_input_img["text"] = self.INPUT_IMG

    def outputImage(self):
        typ = [('PNGファイル','*.png')] 
        # fle = filedialog.asksaveasfilename(filetypes = '*.*', initialdir = dir) 
        fle = filedialog.asksaveasfilename(filetypes = typ) 

        self.OUTPUT_IMG = fle + '.png'
        self.lbl_output_img["text"] = self.OUTPUT_IMG

    def img2img(self):
        try:
            self.onChangeWidthText()
            self.onChangeHeightText()
            self.onChangeStrength()
            self.onChangePrompt()

            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, 
                                    set_alpha_to_one=False) 
            if len(self.INPUT_IMG) > 0:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained( 
                    self.MODEL_ID, 
                    scheduler=scheduler, 
                    # RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'のエラーによりコメントアウト
                    # revision="fp16", 
                    # torch_dtype=torch.float16, 
                    use_auth_token=self.YOUR_TOKEN 
                ).to(self.DEVICE) 

                # pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
                init_img = Image.open(self.INPUT_IMG) 
                init_img = init_img.resize((self.WIDTH, self.HEIGHT)) 
                
                generator = torch.Generator(device=self.DEVICE).manual_seed(self.SEED) 

                # with autocast(DEVICE): 
                #     image = pipe(prompt=PROMPT, init_image=init_img, strength=STRENGTH, guidance_scale=SCALE, 
                #                  generator=generator, num_inference_steps=STEP)["sample"][0] 
                #     image.save("result.png")
                result = pipe(prompt=self.PROMPT, init_image=init_img, strength=self.STRENGTH, guidance_scale=self.SCALE, 
                                generator=generator, num_inference_steps=self.STEP)
                
                image = result.images[0]
                image.save(self.OUTPUT_IMG)

                messagebox.showinfo("成功しました")
            else:
                messagebox.showinfo("パラメータエラー")
        except:
            messagebox.showinfo("失敗しました")

    def txt2img(self):
        try:
            self.onChangeWidthText()
            self.onChangeHeightText()
            self.onChangeStrength()
            self.onChangePrompt()

            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, 
                                    set_alpha_to_one=False) 
            
            # モデルのインスタンス化
            model = StableDiffusionPipeline.from_pretrained(self.MODEL_ID, use_auth_token=self.YOUR_TOKEN)
            model.to(self.DEVICE)

            #prompt = "Tokyo Sky Tree by Marc Chagall" #@param {type:"string"}

            # モデルにpromptを入力し画像生成
            # result = model(self.PROMPT,num_inference_steps=100)["sample"][0]
            result = model(self.PROMPT,num_inference_steps=self.STEP)
            # 保存
            image = result.images[0]
            image.save(self.OUTPUT_IMG)

            messagebox.showinfo("成功しました")
        except:
            messagebox.showinfo("失敗しました")

# Todoアプリを起動する
if __name__ == "__main__":
    app = StableDiffusion()
    app.mainloop()
