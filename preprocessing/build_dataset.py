from datasets import load_from_disk
import random
from src.vocoder import GLM4CodecEncoder, GLM4CodecDecoder
import soundfile as sf
import logging
import os
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize encoder globally
encoder = GLM4CodecEncoder()

system_prompt = "<|system|>\nUser will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

def add_user_input(prompt, user_input):
    return prompt + f"<|user|>\n{user_input}"

def add_assistant_input(prompt, assistant_input):
    return prompt + f"<|assistant|>streaming_transcription\n{assistant_input}"

def remove_overlap_messages(conv):
    conv = sorted(conv, key=lambda x: x['start'])
    for i in range(len(conv)):
        if i == 0:
            continue
        if conv[i]['start'] < conv[i-1]['end']:
            conv[i]['text'] = "element_to_remove"
    conv = [x for x in conv if x['text'] != "element_to_remove"]
    return conv

def merge_same_speaker_messages(conv):
    conv = sorted(conv, key=lambda x: x['start'])
    for i in range(len(conv)):
        if i == 0:
            continue
        if conv[i]['speaker'] == "element_to_remove":
            continue
        if conv[i]['speaker'] == conv[i-1]['speaker']:
            conv[i-1]['text'] += conv[i]['text']
            conv[i]['text'] = "element_to_remove"
            conv[i]['speaker'] = "element_to_remove"
            conv[i-1]['end'] = conv[i]['end']
    conv = [x for x in conv if x['text'] != "element_to_remove"]
    return conv

def text_cleaning(conv):
    for i in range(len(conv)):
        conv[i]['text'] = conv[i]['text'].replace("\"", "")
    return conv

def dump_to_audio(wav, sr, start, end, ch):
    sf.write(f"temp.wav", wav[int(start*sr):int(end*sr), ch].squeeze(), sr)

def process_conversation(conv, audio_path, p_replace_with_text=0.1):
    try:
        conv = remove_overlap_messages(conv)
        conv = merge_same_speaker_messages(conv)
        conv = text_cleaning(conv)

        wav, sr = sf.read(audio_path)

        inputs = "" + system_prompt
        for i in range(len(conv)):
            msg = conv[i]
            if msg['speaker'] == 0:
                if random.random() < p_replace_with_text:
                    inputs = add_user_input(inputs, msg['text'])
                else:
                    dump_to_audio(wav, sr, msg['start'], msg['end'], msg['speaker'])
                    audio_tokens = encoder(["temp.wav"])[0]
                    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
                    inputs = add_user_input(inputs, audio_tokens)
            else:
                inputs = add_assistant_input(inputs, msg['text'])
        
        return inputs
    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}")
        return None

if __name__ == "__main__":
    # Load the dataset
    dataset = load_from_disk("./hf_rpg_overlap_30_35")
    
    # Create a list to store all processed texts
    processed_texts = {"text": []}
    total_conversations = len(dataset)
    
    # Process all conversations
    for idx in range(total_conversations):
        try:
            logger.info(f"Processing conversation {idx+1}/{total_conversations}")
            conv = dataset['conversation'][idx]
            audio_path = os.path.join("/work/u3937558/tp2/_rpg_overlap_30_35/audio", dataset['audio_file'][idx])
            
            processed_text = process_conversation(conv, audio_path)
            if processed_text:
                processed_texts["text"].append(processed_text)
                logger.info(f"Successfully processed conversation {idx+1}")
            else:
                logger.warning(f"Failed to process conversation {idx+1}, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing conversation {idx+1}: {str(e)}")
            continue
    
    # Create and save the new dataset
    from datasets import Dataset
    processed_dataset = Dataset.from_dict(processed_texts)
    processed_dataset.save_to_disk("hf_rpg_overlap_30_35_processed")
    
    logger.info(f"Processing complete. Processed {len(processed_texts)} conversations.")
    logger.info(f"Dataset saved to 'hf_rpg_overlap_30_35_processed'")
