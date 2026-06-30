import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ============== 配置区域 ==============
MODEL_PATH = "Qwen3-VL-4B-Instruct"
IMAGE_BASE_DIR = "/ERIQ/images"
TASK_FILES = [
    "QA_FINE_GRAINED_PLAN.json",
    "QA_SUBTASK_PLANNING.json",
    "QA_TASK_PROGRESS.json",
]
OUTPUT_FILE = "predictions_qwen3.json"
# ======================================


def load_model(model_path):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def build_messages(question_text, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question_text.replace("<image>\n", "").replace("<image>", "") + "Please answer directly with only the letter of the correct option and nothing else."},
            ],
        }
    ]
    return messages


def inference_single(model, processor, messages):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=32)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text.strip()


def run_inference(model_path, image_base_dir, task_files, output_file):
    print(f"Loading model from: {model_path}")
    model, processor = load_model(model_path)
    model.eval()

    all_results = []

    for task_file in task_files:
        print(f"\nProcessing: {task_file}")
        with open(task_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in tqdm(data, desc=task_file):
            item_id = item["id"]
            image_rel_path = item["image"]
            conversations = item["conversations"]
            question_text = conversations[0]["value"]
            gt_answer = conversations[1]["value"]

            image_path = os.path.abspath(os.path.join(image_base_dir, image_rel_path))

            if not os.path.exists(image_path):
                print(f"  [WARNING] Image not found: {image_path}, skipping {item_id}")
                prediction = ""
            else:
                messages = build_messages(question_text, image_path)
                prediction = inference_single(model, processor, messages)
            
            print("id: ",item_id," answer: ",prediction," GT:",gt_answer, " Correct: ", "√" if prediction == gt_answer else "x")

            all_results.append({
                "id": item_id,
                "image": image_rel_path,
                "conversations": conversations,
                "prediction": prediction,
                "gt_answer": gt_answer,
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Total predictions: {len(all_results)}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERIQ Benchmark Inference with Qwen3-VL-4B")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to model weights")
    parser.add_argument("--image_base_dir", type=str, default=IMAGE_BASE_DIR, help="Base directory for images")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output JSON file path")
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        image_base_dir=args.image_base_dir,
        task_files=TASK_FILES,
        output_file=args.output,
    )
