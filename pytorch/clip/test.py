import torch
from PIL import Image
import open_clip
import os
import pandas as pd

in_dir = "/mnt/nis_lab_research/data/clip_data/shah_b1_539_21"
gpu_id = 0
model_name = "ViT-B-16"
pretrain_dataset = "datacomp_l_s1b_b8k"

def main():
    
    torch.cuda.set_device(gpu_id)

    # List pretrained models available in open_clip
    open_clip.list_pretrained()

    # Create the model, tokenizer, and preprocess function
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain_dataset)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Move the model to GPU
    model = model.cuda()

    cats = os.listdir(in_dir)
    acc_list = []
    tp_list = []
    tot_list = []

    for i, cat in enumerate(cats):
        tp_ctr = 0
        cat_pth = os.path.join(in_dir, cat)
        print(cat)
        img_list = [x for x in os.listdir(cat_pth) if x[-3:] == "png"]
        for j, item in enumerate(img_list):
            print("   Image Complete:", j)
            img_pth = os.path.join(cat_pth, item)
            txt_pth = img_pth[:-3] + "txt"
            
            with open(txt_pth, "r") as f:
                cont = f.read()
            
            cat_emb = tokenizer(cats).cuda()
            img_emb = preprocess(Image.open(img_pth)).unsqueeze(0).cuda()
            cont_emb = tokenizer(cont).cuda()
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                cat_feat = model.encode_text(cat_emb)
                img_feat = model.encode_image(img_emb)
                cont_feat = model.encode_text(cont_emb)
                
                # Normalize features
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                cat_feat /= cat_feat.norm(dim=-1, keepdim=True)
                cont_feat /= cont_feat.norm(dim=-1, keepdim=True)
                
                cat_probs_img = (100.0 * img_feat @ cat_feat.T).softmax(dim=-1)[0].cpu().numpy().tolist()
                lab_img = cats[cat_probs_img.index(max(cat_probs_img))]
                
                cat_probs_cont = (100.0 * cont_feat @ cat_feat.T).softmax(dim=-1)[0].cpu().numpy().tolist()
                lab_cont = cats[cat_probs_cont.index(max(cat_probs_cont))]
            
            lab_list = [lab_img, lab_cont]
            max_probs = [max(cat_probs_img), max(cat_probs_cont)]
            fin_lab = lab_list[max_probs.index(max(max_probs))]
            
            if fin_lab == cat:
                tp_ctr += 1
        
        tp_list.append(tp_ctr)
        tot_list.append(len(img_list))
        try:
            acc_list.append(tp_ctr / len(img_list))
        except:
            acc_list.append(0)
        
    res_df = pd.DataFrame({"Categories": cats, "True Positives": tp_list, "Tot Cat Instances": tot_list, "Accuracy": acc_list})
    res_df.to_csv("./clip_results.csv", index=False)

if __name__ == "__main__":
    main()
