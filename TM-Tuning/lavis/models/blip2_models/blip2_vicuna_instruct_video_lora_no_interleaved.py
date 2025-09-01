"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version
import cv2
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np
import transformers
import os
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from peft import LoraConfig, get_peft_model
import lavis.models.blip2_models.Qformer_lora as Qformer_lora 
from lavis.common.utils import is_url
from lavis.common.dist_utils import download_cached_file
from lavis.models.blip2_models.Qformer_lora import lora, custom_lora, mark_only_lora_as_trainable, check_lora_application
from multi_modality.demo.config import (Config,
                    eval_dict_leaf)


#set video mae config
config = Config.from_file('/data/xingjian_luo/project/videollm/internvideo2_stage2_config.py')
config = eval_dict_leaf(config)
model_pth = '/data/xingjian_luo/project/videollm/video_encoder/mp_rank_00_model_states.pt'
config['pretrained_path'] = model_pth
config['origin_num_frames'] = 4
config.model.text_encoder["config"] = "/data/xingjian_luo/project/videollm/multi_modality/configs/config_bert_large.json"
config.model.vision_encoder["num_frames"] = 4
config.model.text_encoder["pretrained"] = "/data/xingjian_luo/checkpoint/bert-large-uncased"



@registry.register_model("blip2_vicuna_instruct_video_lora_no_interleave")
class Blip2VicunaInstructVideoLoRA_NO_INTERLEAVE(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="/data/xingjian_luo/checkpoint/vicuna-7b-v1.1",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        video_cfg = config,
        llm_lora_r=8,
        llm_lora_apply="attn",
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        self.video_cfg = video_cfg
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        #需要重写这个function
        if video_cfg:
            self.video_cfg = video_cfg
            self.visual_encoder, self.intern_tokenizer = self.init_video_encoder(video_cfg)
        else:
            return None
            # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
            # )
        if freeze_vit:
            logging.info("freeze vision encoder")
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            
        #num_features is 768 in ViT, embed_dim=512, clip_embed=768
        #关于feature pooled是B*1*C的综合的embedding, vision_embeds是[B,N,C]，包含每一个frame的embeddings
        if video_cfg:
            print("initalize qformer...")
            #self.video_cfg.model.embed_dim 为512，但其实embedding是1025个1408维的token,d_model=1408
            self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.video_cfg.model.vision_encoder.d_model
        )
        else:
            return None
            # self.Qformer, self.query_tokens = self.init_Qformer(
            #     num_query_token, self.visual_encoder.num_features
            # )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        #print("loading pretrained qformer...")
        # self.Qformer.load_state_dict(torch.load("/data/xingjian_luo/checkpoint/qformer.pth"))
        # Train only the Qformer LoRA
        mark_only_lora_as_trainable(self.Qformer)
        
        check_lora_application(self.Qformer)
        
        num_params = sum([p.numel() for p in self.Qformer.parameters() if p.requires_grad])
        print(f"Number of trainable parameters in Qformer: {num_params}")

        #TODO
        llm_model = "/data/xingjian_luo/checkpoint/vicuna-7b-v1.1"
        print("initialize llm...")
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, load_in_8bit=True, device_map="auto"
        )
        #load_in_8bit=True torch_dtype=torch.float16

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        def _find_all_linear_names(model):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            # if 'lm_head' in lora_module_names: # needed for 16-bit
            #     lora_module_names.remove('lm_head')
            return list(lora_module_names)

        target_modules = []
        if llm_lora_apply == "attn":
            target_modules = ['q_proj','v_proj'] 
        elif llm_lora_apply == "ffn":
            target_modules = ['gate_proj', "up_proj", "down_proj"]
        elif llm_lora_apply == "all":
            target_modules = ['q_proj','v_proj', 'gate_proj', "up_proj", "down_proj"] 
        else: 
            print("Wrong llm_lora_apply value in yaml!!")
        print(f"applying llm lora on {llm_lora_apply}")
        lora_config = LoraConfig(
            r=llm_lora_r,
            lora_alpha=8,
            target_modules=target_modules,
            # lora_dropout=training_args.lora_dropout,
            # bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)
        self.llm_model.print_trainable_parameters()

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def _frame_from_video(self,video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def normalize(self,data):
        v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
        return (data/255.0-v_mean)/v_std


    def frames2tensor(self,vid_list, fnum=4, target_size=(224, 224), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube

    def get_video_tensor(self,vid_tube):
        
        T = vid_tube.shape[1]
        use_image = True if T == 1 else False
        vid_tube = vid_tube.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        vision_embeds, pooled_vision_embeds, _, _  = self.visual_encoder.vision_encoder(vid_tube, None, use_image)

        return vision_embeds,pooled_vision_embeds



    def split_long_video(self,frames,num_frame = config.model.vision_encoder["num_frames"]):
        #split long video into small batches
        video_length = len(frames)
        batches = video_length//num_frame
        frames = frames[:batches*num_frame]
        #get video length
        video_length = len(frames)
        if video_length>= 2*num_frame:
            new_frame_list = [frames[i:i + num_frame] for i in range(0, len(frames), num_frame)]
            return new_frame_list
        else:
            return [frames]
        

    def warp_video_embedding_with_words(self,query_output_list,query_tokens,cur_device):
        #generate a prompt with both video embedding and text tokens
        for i in range(len(query_output_list)):
            if i == 0:
                inputs_embeds = None
                attention_mask = None
                inputs_llm = self.llm_proj(query_output_list[i].last_hidden_state[:,:query_tokens.size(1),:]) #video input embedding
                atts_llm_ = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(cur_device) #video input atts
                inputs_embeds = inputs_llm
                attention_mask = atts_llm_
            else:
                inputs_llm = self.llm_proj(query_output_list[i].last_hidden_state[:,:query_tokens.size(1),:])
                atts_llm_ = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(cur_device)
                inputs_embeds = torch.cat([inputs_embeds, inputs_llm], dim=1)
                attention_mask = torch.cat([attention_mask,atts_llm_], dim=1)
        return inputs_embeds,attention_mask

    
    def add_padding_to_same_size(self,file_inputs_llm,file_inputs_atts,text_input_tokens,text_output_tokens):
        #add padding to the text_input, make them become the same size based on the length of the video
        device = torch.device('cuda')
        length_list = [i.shape[1] for i in file_inputs_llm]
        max_length = max(length_list)
        diff_num = [max_length - i.shape[1] for i in file_inputs_llm]
        new_inputs_embeds = None
        new_inputs_masks = None
        total_targets = None
        for index,n in enumerate(diff_num):
            if n > 0:
                padding_tensor = torch.full((n,),32000,dtype=torch.long).to(device)
                text_input_tokens_input_ids = torch.cat([text_input_tokens.input_ids[index],padding_tensor],dim=0)
                zero_mask_tensor = torch.full((n,),0,dtype=torch.long).to(device)
                text_input_tokens_attention_mask = torch.cat([text_input_tokens.attention_mask[index],zero_mask_tensor],dim=0)
            else:
                text_input_tokens_input_ids = text_input_tokens.input_ids[index]
                text_input_tokens_attention_mask = text_input_tokens.attention_mask[index]

            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens_input_ids.unsqueeze(0),
                text_input_tokens_attention_mask.unsqueeze(0),
                text_output_tokens.input_ids[index].unsqueeze(0),
                text_output_tokens.attention_mask[index].unsqueeze(0),
            )
            inputs_embeds_ = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = torch.cat([file_inputs_llm[index],inputs_embeds_],dim=1)
            inputs_masks = torch.cat([file_inputs_atts[index],llm_tokens["attention_mask"]],dim=1)
            if new_inputs_embeds is None:
                new_inputs_embeds = inputs_embeds
                new_inputs_masks = inputs_masks
            else:
                new_inputs_embeds = torch.cat([new_inputs_embeds,inputs_embeds],dim=0)
                new_inputs_masks = torch.cat([new_inputs_masks,inputs_masks],dim=0)
                
            #do not apply loss to paddings 
            targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
            )
            # do not apply loss to the query tokens
            empty_targets = (
                torch.ones(file_inputs_atts[index].size(), dtype=torch.long).to(device).fill_(-100)
            )
            # do not apply loss to the text input (i.e., instruction)
            targets[0][:input_part_targets_len[0]] = -100

            targets = torch.cat([empty_targets, targets], dim=1)

            if total_targets is None:
                total_targets = targets
            else:
                total_targets = torch.cat([total_targets,targets],dim=0)

        return new_inputs_embeds,new_inputs_masks,total_targets

    
    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')
        #先提取到视频片段
        if self.video_cfg:
            cur_device = torch.device('cuda')
            video_paths = samples["video"] #should be a list of videos
            text_Qformer = self.tokenizer(
                            samples["text_input"],
                            padding='longest',
                            truncation=True,
                            max_length=self.max_txt_len,
                            return_tensors="pt",
                        ).to(cur_device)

            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'
            text_input_tokens = self.llm_tokenizer(
                samples['text_input'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(cur_device)

            self.llm_tokenizer.truncation_side = 'right'
            text_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(cur_device)


            self.llm_model.to(cur_device)

            # llm_tokens, input_part_targets_len = self.concat_text_input_output(
            #     text_input_tokens.input_ids,
            #     text_input_tokens.attention_mask,
            #     text_output_tokens.input_ids,
            #     text_output_tokens.attention_mask,
            # )
            # inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

            self.llm_proj.to(cur_device)


            file_inputs_llm = []
            file_inputs_atts = []


            for index,path in enumerate(video_paths):
                video = cv2.VideoCapture(path)
                frames = [x for x in self._frame_from_video(video)]
                frame_list = self.split_long_video(frames)
                query_output_list = []
                for frame in frame_list:

                    #需要设置一个PADDING机制使得最终长度一致
                    vid_tubes = None
                    vid_tube = self.frames2tensor(frame)
                    if vid_tubes is None:
                        vid_tubes = vid_tube
                    else:
                        vid_tubes = torch.cat((vid_tubes,vid_tube),dim=0)
                
                    with self.maybe_autocast():
                        #video_embeds : [1, 1025, 1408], pooled : [1, 768] pooled是经过了一个attention池化层，把query做了一个平均，用一个平均的query去做attention，得到一个一维的维度
                        video_embeds,pooled_vision_embeds = self.get_video_tensor(vid_tubes)
                    # print(video_embeds.shape)
                    #TODO adaptive pooling
                    #adaptive_pool = nn.AdaptiveAvgPool1d(256)
            


                    video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(vid_tube.device)
                    bs = vid_tube.size(0) #batch size of videos
                    cur_device = vid_tube.device
                    self.Qformer.bert.to(cur_device)
                    query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1).to(cur_device)
                    if self.qformer_text_input:
                        
                        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(cur_device)
                        #print("query_atts: ",query_atts.shape)
                        #print("text_Qformer.attention_mask: ",text_Qformer.attention_mask.shape)
                        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask[index].unsqueeze(0)],dim=1)

                        query_output = self.Qformer.bert(
                            text_Qformer.input_ids[index].unsqueeze(0),
                            attention_mask=Qformer_atts,
                            query_embeds=query_tokens,
                            encoder_hidden_states=video_embeds,
                            encoder_attention_mask=video_atts,
                            return_dict=True,
                        )
                        query_output_list.append(query_output)
                    else:
                        query_output = self.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=video_embeds,
                            encoder_attention_mask=video_atts,
                            return_dict=True,
                        )
                        query_output_list.append(query_output)

                inputs_llm,atts_llm = self.warp_video_embedding_with_words(query_output_list,query_tokens,cur_device)
                
                file_inputs_llm.append(inputs_llm)
                file_inputs_atts.append(atts_llm)
                




        # inputs_embeds = torch.cat([file_inputs_llm, inputs_embeds], dim=1)
        # attention_mask = torch.cat([file_inputs_atts, llm_tokens['attention_mask']], dim=1)
        inputs_embeds,attention_mask,targets = self.add_padding_to_same_size(file_inputs_llm,file_inputs_atts,text_input_tokens,text_output_tokens)

        #print("inputs_embeds.shape: ",inputs_embeds.shape)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"
        
        if self.video_cfg:
            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            video_paths = samples["video"] #should be a list of videos
            
            for path in video_paths:
                video = cv2.VideoCapture(path)
                frames = [x for x in self._frame_from_video(video)]
                frame_list = self.split_long_video(frames)
                query_output_list = []
                for frame in frame_list:
                    vid_tubes = None
                    vid_tube = self.frames2tensor(frame)
                    if vid_tubes is None:
                        vid_tubes = vid_tube
                    else:
                        vid_tubes = torch.cat((vid_tubes,vid_tube),dim=0)
                
                    with self.maybe_autocast():
                        #video_embeds : [1, 1025, 1408], pooled : [1, 768] pooled是经过了一个attention池化层，把query做了一个平均，用一个平均的query去做attention，得到一个一维的维度
                        video_embeds,pooled_vision_embeds = self.get_video_tensor(vid_tubes)
                    # print(video_embeds.shape)
                    video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(vid_tube.device)

                    
                    bs = vid_tube.size(0) #batch size of videos
                    cur_device = vid_tube.device
                    self.Qformer.bert.to(cur_device)
                    if isinstance(prompt, str):
                        prompt = [prompt] * bs
                    else:
                        assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

                    # For TextCaps
                    if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
                        prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

                    query_tokens = self.query_tokens.expand(bs, -1, -1).to(cur_device)
                    if self.qformer_text_input:
                        text_Qformer = self.tokenizer(
                            prompt,
                            padding='longest',
                            truncation=True,
                            max_length=self.max_txt_len,
                            return_tensors="pt",
                        ).to(cur_device)
                        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(cur_device)
                        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                        
                    video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(cur_device)
                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=video_embeds,
                        encoder_attention_mask=video_atts,
                        return_dict=True,
                    )
                    query_output_list.append(query_output)

        else:
            return None

        self.llm_proj.to(cur_device)
        # inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        # atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(cur_device)
        inputs_llm,atts_llm = self.warp_video_embedding_with_words(query_output_list,query_tokens,cur_device)

        if self.video_cfg:
            llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
            ).to(cur_device)
        else:
            return None
            # llm_tokens = self.llm_tokenizer(
            #     prompt,
            #     padding="longest",
            #     return_tensors="pt"
            # ).to(cur_device)

        self.llm_model.to(cur_device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text


    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        llm_lora_r = cfg.get("llm_lora_r", 8)
        llm_lora_apply = cfg.get("llm_lora_apply", "attn") 
        
        r = cfg.get("lora_r", 8)
        alpha = cfg.get("lora_alpha", 16)
        dropout = cfg.get("lora_dropout", 0.05)
        
        self_attention_qv_lora = cfg.get("self_attention_qv_lora", False)
        self_attention_output_lora = cfg.get("self_attention_output_lora", False)
        ffn_lora = cfg.get("ffn_lora", False)
        
        qformer_crossattention_lora_q = cfg.get("qformer_crossattention_lora_q", False)
        qformer_crossattention_lora_k = cfg.get("qformer_crossattention_lora_k", False)
        qformer_crossattention_lora_v = cfg.get("qformer_crossattention_lora_v", False)
        qformer_crossattention_lora_o = cfg.get("qformer_crossattention_lora_o", False)
        with lora(r, alpha, dropout, enabled=self_attention_qv_lora, qkv=[qformer_crossattention_lora_q, qformer_crossattention_lora_k, qformer_crossattention_lora_v]), custom_lora(r, alpha, dropout, enabled=(self_attention_output_lora or qformer_crossattention_lora_o), type="BertSelfOutput", sc=[self_attention_output_lora, qformer_crossattention_lora_o]), custom_lora(r, alpha, dropout, enabled=ffn_lora, type="BertOutput"):
            model = cls(
                vit_model=vit_model,
                img_size=img_size,
                drop_path_rate=drop_path_rate,
                use_grad_checkpoint=use_grad_checkpoint,
                vit_precision=vit_precision,
                freeze_vit=freeze_vit,
                num_query_token=num_query_token,
                llm_model=llm_model,
                prompt=prompt,
                max_txt_len=max_txt_len,
                max_output_txt_len=max_output_txt_len,
                apply_lemmatizer=apply_lemmatizer,
                qformer_text_input=qformer_text_input,
                llm_lora_r=llm_lora_r,
                llm_lora_apply=llm_lora_apply
            )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model


    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # strict=False for peft layers
        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
