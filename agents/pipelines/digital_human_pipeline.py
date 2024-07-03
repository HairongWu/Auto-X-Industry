import numpy as np
import re
from transformers import AutoModel, AutoTokenizer
from StyleTTS2.styletts2 import StyleTTS2

from scipy.io import wavfile
from EAT.demo import EAT
from EAT.preprocess.deepspeech_features.extract_ds_features import extract_features

import whisper

from .pipeline import *

def add_quotes_if_missing(input_string):
    search_list = ['"Empathetic Response":', '"Emotional Response":']
    quote = "\""
    output_string = input_string
    # 查找目标字符串
    for search_string in search_list:
        index = output_string.find(search_string)
        if index != -1:
            # 判断目标字符串后面是否有双引号
            if output_string[index + len(search_string)] != quote:
                # 在目标字符串后面添加双引号
                output_string = output_string[:index + len(search_string)] + quote + output_string[index + len(search_string):]
            if output_string[index-1] != quote:
                output_string = output_string[:index-1] + quote + ',' + output_string[index:]
    return output_string

emotion_aligned = {
                    "suprised":"sur",
                    "excited":"hap",
                    "angry":"ang",
                    "proud":"hap",
                    "sad":"sad",
                    "annoyed":"ang",
                    "grateful":"hap",
                    "lonely":"sad",
                    "afraid":"fea",
                    "terrified":"fea",
                    "guilty":"sad",
                    "impressed":"sur",
                    "disgusted":"dis",
                    "hopeful":"hap",
                    "confident":"neu",
                    "furious":"ang",
                    "anxious":"sad",
                    "anticipating":"hap",
                    "joyful":"hap",
                    "nostalgic":"sad",
                    "disappointed":"sad",
                    "prepared":"neu",
                    "jealous":"ang",
                    "content":"hap",
                    "devastated":"sur",
                    "embarrassed":"neu",
                    "caring":"hap",
                    "sentimental":"sad",
                    "trusting":"neu",
                    "ashamed":"neu",
                    "apprehensive":"fea",
                    "faithful":"neu",
}
                
prompt = "You are an experienced and empathetic chatbot. Unlike regular chat systems, you are a virtual digital entity that adapts its gender, age, and voice to better serve different users. Now you are provided with a conversation which contains user's current queries and conversation history.Your task is firstly to comprehend the provided conversation information to the best of your ability, and then to proceed with a step-by-step, in-depth analysis following the procedure outlined below, and finally output your results of each step.Step 1: <Emotion Cause> The reason behind the user's emotional state.Step 2: <Event Scenario> The scenario in which the conversation takes place. You can focus on the events mentioned in the conversation or infer the scenario based on common sense and domain knowledge, such as daily conversation, psychological assistance, elder people company, or children company, etc. The result of this step is summary-oriented, ideally consisting of no more than 5 words and allowing for repetition.Step 3: <Rationale> The underlying reasons behind the user's emotions or the occurrence of the current event.Step 4: <Goal to Response> Determine the goal the chatbot should aim for in its response based on the analysis from the previous steps.Step 5: <Agent gender> The gender of the chatbot, selecting from male and female.Step 6: <Agent age> The age of the chatbot, selecting from Children, Teenagers, Young Adults, Middle-aged Adults and Elderly Adults.Step 7: <Agent Timbre and Tone> The timbre of the chatbot, selecting from Low-pitched, Soft, Clear, Melodious, warm, husky, bright.Step 8: <Empathetic Response> Leveraging the comprehensive analysis of the aforementioned steps, provide the user with an empathetic response to their query.Step 9: <Emotional Response> The emotional tone conveyed when replying to the user.Here is an example, given the input: {'User Query': \"It's all because of the traffic jam. It was terrible and very frustrating.\", 'Conversation History': '[User]I was late for work today.[Agent]Can you tell me what happened?'}. Your output must strictly comply with the JSON format and refrain from outputting any other irrelevant content, as shown in the following output example: {'Emotion Cause': 'Traffic jam.', 'Event Scenario': 'Work-related stress.', 'Rationale': 'Traffic jam results in lateness, causing individuals to feel anxious and frustrated.', 'Goal to Response': 'Alleviating anxiety and agitation.', 'Agent Gender': 'Female', 'Agent Age': 'Young adults', 'Agent Timbre and Tone': 'Soft', 'Empathetic Reponse': \"I can imagine how frustrating and challenging it must have been to deal with such a terrible traffic jam. It's understandable that it caused you to be late for work.\", 'Emotional Response': 'Terrified'}. \n"

class DigitalHumanPipeline(Pipeline): 
    def __init__(self):
        super().__init__()

        self.whisper_model = whisper.load_model("medium")

        self.chatglm_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        self.chatglm_model = AutoModel.from_pretrained("THUDM/chatglm3-6b", load_in_8bit=False, trust_remote_code=True, device_map="auto")
        self.max_new_token = 2048

    def predict(self, audio_file):
        results = []
        
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # detect the spoken language
        _, probs = self.whisper_model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(self.whisper_model, mel, options)

        # print the recognized text
        print(result.text)

        cur_prompt = prompt + f"Now here is an input for you:{result.text}"

        chatglm_inputs = self.chatglm_tokenizer(cur_prompt, return_tensors="pt").to(self.device)
        response = self.chatglm_model.generate(input_ids=chatglm_inputs["input_ids"],
                            max_length=chatglm_inputs["input_ids"].shape[-1] + self.max_new_tokens)
        response = response[0, chatglm_inputs["input_ids"].shape[-1]:]
        text_response = self.chatglm_tokenizer.decode(response, skip_special_tokens=True)
        print(text_response)

        try:
            # # 移除首尾的引号
            text_response = add_quotes_if_missing(text_response)

            # 将字符串解析为字典
            pattern =  r'"([^"]+)"\s*:\s*([^,]+)'
            matches = re.findall(pattern, text_response)
            data = {key: value for key, value in matches}
            emotion_cause = data["Emotion Cause"].strip('"')
            event_scenario =  data["Event Scenario"].strip('"')
            rationale = data["Rationale"].strip('"')
            goal_to_response = data["Goal to Response"].strip('"')
            agent_gender = data["Agent Gender"].strip('"')
            agent_age = data["Agent Age"].strip('"')
            agent_timbre_tone = data["Agent Timbre and Tone"].strip('"')
            empathetic_response = data["Empathetic Response"].strip('"')
            if "Emotional Response" in data:
                emotional_response = data["Emotional Response"].strip('"')
                try:
                    emotion_type = emotion_aligned[emotional_response]
                except:
                    emotion_type = "neu"
            else:
                emotional_response = "neu"
        except:
            return

        #TTS for response text
        tts = StyleTTS2()
        if agent_gender=='Female':
            wav_file = "StyleTTS2/Demo/reference_audio/W/" + agent_timbre_tone.lower() + '.wav'
        elif agent_gender=='Male':
            wav_file = "StyleTTS2/Demo/reference_audio/M/" + agent_timbre_tone.lower() + '.wav'
        # Wav_path = "StyleTTS2/Demo/EED_wav/"
        # Wav_dict = get_all_filenames(Wav_path)
        # for wav_file in Wav_dict:
        result_name = wav_file.split('/')[-1]

        ref_s = tts.compute_style(wav_file)
        wav = tts.inference(empathetic_response, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)

        scaled_data = np.int16(wav * 32767)
        wav_file_path = wav_save_path + '/' + result_name
        wavfile.write(wav_file_path, 24000, scaled_data)


        #extract wav deepspeech features
        extract_features(wav_save_path, wav_save_path + '/deepfeature32')
        #wav2talkingface
        eat = EAT(root_wav=wav_save_path)
        eat.tf_generate(agent_age, agent_gender, emotion_type, save_dir=mp4_save_path)

        return results
