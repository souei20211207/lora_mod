# coding: utf-8
import safetensors
from safetensors.torch import load_file,save_file
import torch
import sys

def help_comment():
    print("py lora_mod.py input_file output_file\ninput_file is safetensors file.\noutput_file is safetensors file.If you don't enter,input_file is overwritten.")
    sys.exit()
    
args = sys.argv

ok=[
    "lora_unet_out_2.alpha",
    "lora_unet_out_2.lora_down.weight",
    "lora_unet_out_2.lora_up.weight",
    "lora_unet_input_blocks_0_0.alpha",
    "lora_unet_input_blocks_0_0.lora_down.weight",
    "lora_unet_input_blocks_0_0.lora_up.weight",
    "lora_unet_label_emb_0_0.alpha",
    "lora_unet_label_emb_0_0.lora_down.weight",
    "lora_unet_label_emb_0_0.lora_up.weight",
    "lora_unet_label_emb_0_2.alpha",
    "lora_unet_label_emb_0_2.lora_down.weight",
    "lora_unet_label_emb_0_2.lora_up.weight",
    "lora_unet_time_embed_0.alpha",
    "lora_unet_time_embed_0.lora_down.weight",
    "lora_unet_time_embed_0.lora_up.weight",
    "lora_unet_time_embed_2.alpha",
    "lora_unet_time_embed_2.lora_down.weight",
    "lora_unet_time_embed_2.lora_up.weight"
]

if len(args)<2:
    help_comment()
else:
    if args[1].endswith(".safetensors"):
        in_path=args[1]
    else:
        help_comment()
if len(args)<3:
    out_path=args[1]
else:
    if args[2].endswith(".safetensors"):
        out_path=args[2]
    else:
        help_comment()
state_dict = load_file(in_path)
ks=[]
ws=[]
for k, w in state_dict.items():
    if not(k in ok):
        ks.append(k)
        ws.append(w)

l=len(ks)
state_dict={}
for i in range(l):
    state_dict[ks[i]]=ws[i]

save_file(state_dict,out_path)
